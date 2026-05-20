"""
Phase R: causal test of encoder action-conditioning trade-off.

Phase N showed that the vae_s1 encoder is severely action-blind
(action_dep_ratio = 0.0003) while vae_s2/s3 are usable. We hypothesize that
this trade-off is causal: encoder training that maximizes probe-class
accuracy (or just standard VAE recon + KL) selects against preserving
action-distinguishing information in the latent.

This script fine-tunes the vae_s1 encoder with an auxiliary "action decoder"
loss that explicitly preserves action information:

    z_t     = encoder(obs_t)
    z_{t+1} = encoder(obs_{t+1})
    L_aux   = CE(action_decoder(z_t, z_{t+1}), a_t)

    L_total = L_recon + β·L_kl + λ·L_aux

Sweeping λ traces a (probe_accuracy, action_dep_ratio) trade-off. If the
trade-off is monotonic anti-correlated, the causal claim is established:
probe accuracy is selected against by action-information preservation.

Usage:
    python finetune_encoder_action_cond.py \\
        --model_path /path/to/sweep_dk8_vae_s1_best_model.pt \\
        --output_path /path/to/finetuned_lam{LAMBDA}.pt \\
        --action_aux_weight {LAMBDA} --epochs 10
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env  # noqa: E402
from analyze_wm_multistep import load_checkpoint  # noqa: E402


# ── Action decoder head ──────────────────────────────────────────────────

class ActionDecoder(nn.Module):
    """Predicts a_t from (z_t, z_{t+1}). Small MLP."""
    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_tp1], dim=-1))


# ── Buffer collection ────────────────────────────────────────────────────

def collect_obs_buffer(env, ae_model, policy, n_random_eps, n_policy_eps,
                       max_steps, device, trans_kind='continuous'):
    """Collect raw (obs_t, a_t, obs_{t+1}) tuples. We don't pre-encode here —
    we'll re-encode through the FINE-TUNED encoder during training. For VQ-VAE
    we must request the post-quantization continuous embedding (not the
    default Long code indices) so the trained policy receives the same input
    distribution it saw during training."""
    obs_in, obs_out, actions = [], [], []
    n_total = n_random_eps + n_policy_eps
    for ep_idx in range(n_total):
        use_policy = (ep_idx >= n_random_eps) and (policy is not None)
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        for step in range(max_steps):
            if use_policy:
                with torch.no_grad():
                    obs_t_dev = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    if trans_kind == 'discrete':
                        # Get post-quantization continuous embedding via STE.
                        encoder_out = ae_model.encoder(obs_t_dev)
                        _qloss, quantized, _, _ = ae_model.quantizer(encoder_out)
                        z_for_pol = quantized.reshape(1, -1)
                    else:
                        z_for_pol = ae_model.encode(obs_t_dev).view(1, -1)
                    a = int(policy(z_for_pol).argmax(dim=-1).item())
            else:
                a = env.action_space.sample()
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            done = sr[2] if len(sr) >= 3 else False
            obs_in.append(obs.astype(np.float32))
            actions.append(int(a))
            obs_out.append(obs_next.astype(np.float32))
            obs = obs_next
            if done:
                break
        if (ep_idx + 1) % 50 == 0:
            print(f'  collected {ep_idx+1}/{n_total} eps  '
                  f'transitions={len(actions)}')
    return (np.stack(obs_in), np.array(actions, dtype=np.int64),
            np.stack(obs_out))


class TripleDataset(Dataset):
    def __init__(self, obs_in, actions, obs_out):
        self.obs_in  = obs_in
        self.actions = actions
        self.obs_out = obs_out
    def __len__(self): return len(self.actions)
    def __getitem__(self, i):
        return (self.obs_in[i], self.actions[i], self.obs_out[i])


# ── Fine-tuning loop ─────────────────────────────────────────────────────

def vae_forward(ae_model, obs):
    """Forward pass through a continuous spatial VAE encoder. Returns
    (z, recon, mu, log_sigma) where z is the flat latent. Uses the same
    encode/decode interface that AEModelSpatial exposes; KL is computed
    from mu/sigma if available, otherwise zero."""
    if hasattr(ae_model, 'encode_dist'):
        # VAE: returns mu, sigma
        mu, sigma = ae_model.encode_dist(obs)
        z = mu + sigma * torch.randn_like(sigma)
        recon = ae_model.decode(z)
        log_sigma = torch.log(sigma + 1e-8)
        return z.view(z.shape[0], -1), recon, mu, log_sigma
    # Fallback: deterministic AE
    z = ae_model.encode(obs)
    recon = ae_model.decode(z)
    return z.view(z.shape[0], -1), recon, torch.zeros_like(z), torch.zeros_like(z)


def vq_forward(ae_model, obs):
    """Forward pass through a VQ-VAE encoder. Returns (z_flat, recon, vq_loss).
    The post-quantization embeddings have STE gradient flow back to the
    encoder; the auxiliary action loss can therefore reshape encoder outputs
    even though the bottleneck is discrete."""
    encoder_out = ae_model.encoder(obs)
    quantizer_loss, quantized, perplexity, oh = ae_model.quantizer(encoder_out)
    recon = ae_model.decoder(quantized)
    z_flat = quantized.reshape(quantized.shape[0], -1)
    return z_flat, recon, quantizer_loss


def finetune(args):
    device = args.device
    print(f'Loading source ckpt: {args.model_path}')
    ae_model, _trans_orig, trans_kind, env, margs = load_checkpoint(args.model_path, device)
    if trans_kind not in ('continuous', 'discrete'):
        raise NotImplementedError(f'Unsupported trans_kind={trans_kind}.')
    ae_model = ae_model.to(device).train()
    print(f'  trans_kind={trans_kind}; using {"VQ" if trans_kind == "discrete" else "VAE"} forward path')
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    input_dim = L * embedding_dim
    n_actions = env.action_space.n

    # Load original policy (for non-uniform episode collection); on failure
    # we fall back to random actions only.
    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    policy = None
    if 'policy_state_dict' in src and 'args' in src:
        try:
            from analyze_wm_phaseB_e8 import build_mlp
            policy = build_mlp(input_dim, src['args'].get('policy_hidden', [256, 256]),
                               n_actions, 'relu').to(device).eval()
            policy.load_state_dict(src['policy_state_dict'])
        except Exception as e:
            print(f'  (policy load failed: {e}; using random-only buffer)')
            policy = None

    print(f'\n[1/3] Collecting buffer ({args.n_random_eps} random + {args.n_policy_eps} policy eps)')
    obs_in, actions, obs_out = collect_obs_buffer(
        env, ae_model, policy, args.n_random_eps, args.n_policy_eps,
        args.max_steps, device, trans_kind=trans_kind)
    print(f'  buffer: {len(actions)} transitions')

    ds = TripleDataset(obs_in, actions, obs_out)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Action decoder
    action_decoder = ActionDecoder(input_dim, n_actions,
                                   hidden=args.action_decoder_hidden).to(device)

    # Optimize encoder + action decoder; keep policy frozen
    params = list(ae_model.parameters()) + list(action_decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    print(f'\n[2/3] Fine-tuning λ={args.action_aux_weight:.3f} for {args.epochs} epochs')
    print(f'  L = recon + {args.kl_weight}·KL + {args.action_aux_weight}·CE(action_decoder)')
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_recon, ep_kl, ep_aux, ep_acc, n_b = 0.0, 0.0, 0.0, 0.0, 0
        for batch in dl:
            obs_t, a_t, obs_tp1 = [b.to(device) for b in batch]
            obs_t   = obs_t.float()
            obs_tp1 = obs_tp1.float()
            a_t     = a_t.long()

            if trans_kind == 'discrete':
                z_t,   recon_t,   vq_t   = vq_forward(ae_model, obs_t)
                z_tp1, recon_tp1, vq_tp1 = vq_forward(ae_model, obs_tp1)
                recon_loss = F.mse_loss(recon_t, obs_t) + F.mse_loss(recon_tp1, obs_tp1)
                kl_loss = vq_t + vq_tp1  # repurpose kl_loss slot for VQ commitment loss
            else:
                z_t,   recon_t,   mu_t,   logs_t   = vae_forward(ae_model, obs_t)
                z_tp1, recon_tp1, mu_tp1, logs_tp1 = vae_forward(ae_model, obs_tp1)
                recon_loss = F.mse_loss(recon_t, obs_t) + F.mse_loss(recon_tp1, obs_tp1)
                kl_t = (-0.5 * (1 + 2 * logs_t - mu_t**2 - (2 * logs_t).exp()).sum(dim=tuple(range(1, mu_t.dim())))).mean()
                kl_tp1 = (-0.5 * (1 + 2 * logs_tp1 - mu_tp1**2 - (2 * logs_tp1).exp()).sum(dim=tuple(range(1, mu_tp1.dim())))).mean()
                kl_loss = kl_t + kl_tp1

            logits = action_decoder(z_t, z_tp1)
            aux_loss = F.cross_entropy(logits, a_t)
            aux_acc = (logits.argmax(dim=-1) == a_t).float().mean()

            loss = recon_loss + args.kl_weight * kl_loss + args.action_aux_weight * aux_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            ep_recon += recon_loss.item(); ep_kl += kl_loss.item()
            ep_aux += aux_loss.item(); ep_acc += aux_acc.item(); n_b += 1
        elapsed = time.time() - t0
        print(f'  epoch {epoch+1:>3}/{args.epochs}  recon={ep_recon/n_b:.4f}  '
              f'kl={ep_kl/n_b:.4f}  aux={ep_aux/n_b:.4f}  '
              f'aux_acc={ep_acc/n_b:.3f}  t={elapsed:.1f}s')

    # ── Save the fine-tuned encoder in source-ckpt format ────────────────
    print(f'\n[3/3] Saving fine-tuned encoder to {args.output_path}')
    out = copy.deepcopy(src)
    # Replace encoder weights only
    ae_state = ae_model.state_dict()
    if 'encoder_state_dict' in out:
        out['encoder_state_dict'] = {k: v.cpu() for k, v in ae_state.items()}
    elif 'ae_model_state_dict' in out:
        out['ae_model_state_dict'] = {k: v.cpu() for k, v in ae_state.items()}
    elif 'ae_state_dict' in out:
        out['ae_state_dict'] = {k: v.cpu() for k, v in ae_state.items()}
    else:
        # Not sure which key — search for AE-shaped keys
        for k in list(out.keys()):
            if 'ae' in k.lower() or 'encoder' in k.lower():
                out[k] = {kk: vv.cpu() for kk, vv in ae_state.items()}
                print(f'  (saved encoder weights into {k})')
                break
        else:
            # Fallback: store under a new key plus the canonical one
            out['ae_model_state_dict'] = {k: v.cpu() for k, v in ae_state.items()}
    out['phase_R_aux_weight'] = args.action_aux_weight
    out['phase_R_aux_acc_final'] = float(ep_acc / max(n_b, 1))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(out, args.output_path)
    print(f'Saved.  Final aux action acc: {ep_acc / max(n_b, 1):.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Source vae_s1 ckpt')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--action_aux_weight', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_random_eps', type=int, default=200)
    parser.add_argument('--n_policy_eps', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--action_decoder_hidden', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    finetune(args)


if __name__ == '__main__':
    main()
