"""
Phase G: train an offline-to-convergence WM with a *multi-step* prediction loss.

Standard offline WM training (Phase E2 / train_oracle_wm.py) optimises a 1-step
loss: predict z*_{t+1} given (z*_t, a_t). That trains the WM to be accurate when
fed real states, but does not train it to be self-consistent over a free-running
rollout. Phase J showed that the resulting oracle WM has v_corr +0.88 at k=1 but
drops to +0.57 at k=10 — the WM drifts off the encoder manifold over horizon.

Phase G adds an explicit multi-step loss: for each starting state, roll the WM
forward K steps using its OWN predictions, and compute MSE against the true
state at every intermediate step. Gradients flow through the entire rollout
chain, training the WM to be self-consistent. This is the standard Dreamer/RSSM
multi-step training scheme adapted to our offline-on-frozen-encoder setting.

VAE-only in this first version (clean continuous gradients). VQ extension would
need straight-through estimators on the codebook lookup; left to follow-up.

Usage:
    python train_multistep_oracle_wm.py \\
        --model_path src.pt --output_path multistep.pt \\
        --K 5 --epochs 100 --n_random_eps 300 --n_policy_eps 300
"""
import argparse
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

from analyze_wm_phaseB_e8 import (  # noqa: E402
    build_mlp, encode_flat,
)


def collect_traj_buffer(env, ae_model, policy, n_random_eps, n_policy_eps,
                        max_steps, K, device, trans_kind):
    """Collect trajectories sliced into (z_t, a_t..a_{t+K-1}, z_{t+1..t+K}) tuples.
    Mix of random + policy actions. Each trajectory must be at least K+1 long."""
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L
    n_total = n_random_eps + n_policy_eps

    z0_list, action_list, z_target_list = [], [], []

    for ep in range(n_total):
        use_policy = (ep >= n_random_eps) and (policy is not None)
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        z_seq = [encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy().astype(np.float32)]
        a_seq = []
        for step in range(max_steps):
            if use_policy:
                with torch.no_grad():
                    logits = policy(torch.from_numpy(z_seq[-1]).unsqueeze(0).to(device))
                    a = int(logits.argmax(dim=-1).item())
            else:
                a = env.action_space.sample()
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            done = sr[2] if len(sr) >= 3 else False
            obs_t = torch.from_numpy(obs_next).float()
            z_seq.append(encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy().astype(np.float32))
            a_seq.append(a)
            if done:
                break

        # Slice into starting points where we have K future steps available
        for t in range(len(a_seq) - K + 1):
            z0_list.append(z_seq[t])
            action_list.append(np.array(a_seq[t:t+K], dtype=np.int64))
            z_target_list.append(np.stack(z_seq[t+1:t+K+1]))

        if (ep + 1) % 50 == 0:
            print(f'  buffer: {ep+1}/{n_total} eps, {len(z0_list)} K={K}-step samples')

    return {
        'z0':       np.stack(z0_list).astype(np.float32),         # (N, D*L)
        'actions':  np.stack(action_list),                         # (N, K) long
        'z_targets': np.stack(z_target_list).astype(np.float32),   # (N, K, D*L)
    }


class MultiStepDataset(Dataset):
    def __init__(self, buf):
        self.z0 = buf['z0']
        self.actions = buf['actions']
        self.z_targets = buf['z_targets']

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return self.z0[i], self.actions[i], self.z_targets[i]


def train_multistep_continuous(buf, env, ae_model, embedding_dim,
                               hidden, depth, K, device,
                               epochs, batch_size, lr,
                               value_head=None, value_align_coef=0.0):
    """Train ContinuousTransitionModel with multi-step state loss; optionally
    also a value-alignment loss using the FROZEN trained value head V_eta.

    value_align_coef: weight on the value-alignment term (set 0 to disable;
        non-zero requires `value_head` to be passed).

    Per-step loss at horizon k:
        state: MSE(ẑ_k, z*_k)
        value (if enabled): MSE(V_eta(ẑ_k), V_eta(z*_k).detach())
    Total loss is the per-k sum/K of state + value_align_coef * value.
    """
    from shared.models.transition_models import ContinuousTransitionModel
    L = ae_model.n_latent_embeds
    input_dim = L * embedding_dim

    trans_model = ContinuousTransitionModel(
        input_dim, env.action_space,
        hidden_sizes=[hidden] * depth,
    ).to(device)
    trans_model.train()

    ds = MultiStepDataset(buf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=0)
    opt = torch.optim.Adam(trans_model.parameters(), lr=lr)

    use_va = value_head is not None and value_align_coef > 0
    if use_va:
        for p in value_head.parameters():
            p.requires_grad = False
        value_head.eval()
        print(f'  + value-alignment loss enabled (coef={value_align_coef})')

    for ep in range(epochs):
        ep_state_loss_per_k = [0.0] * K
        ep_value_loss_per_k = [0.0] * K
        ep_n = 0
        for z0, actions, z_targets in loader:
            z0 = z0.float().to(device)            # (B, D*L)
            actions = actions.long().to(device)   # (B, K)
            z_targets = z_targets.float().to(device)  # (B, K, D*L)

            cur = z0
            state_losses, value_losses = [], []
            for k in range(K):
                a_k = actions[:, k]
                preds, _, _ = trans_model(cur, a_k)
                target = z_targets[:, k]
                ls = F.mse_loss(preds, target)
                state_losses.append(ls)
                if use_va:
                    with torch.no_grad():
                        v_target = value_head(target)
                    v_pred = value_head(preds)
                    lv = F.mse_loss(v_pred, v_target)
                    value_losses.append(lv)
                cur = preds  # free-running with gradient flow

            loss = sum(state_losses) / K
            if use_va:
                loss = loss + value_align_coef * (sum(value_losses) / K)

            opt.zero_grad()
            loss.backward()
            opt.step()

            for k in range(K):
                ep_state_loss_per_k[k] += float(state_losses[k].item()) * z0.shape[0]
                if use_va:
                    ep_value_loss_per_k[k] += float(value_losses[k].item()) * z0.shape[0]
            ep_n += z0.shape[0]

        if (ep + 1) % 10 == 0 or ep == 0:
            line = f'  ep {ep+1:3d}/{epochs}  state ' + ' '.join(
                [f'k={k+1}:{ep_state_loss_per_k[k]/ep_n:.4f}' for k in range(K)])
            if use_va:
                line += '  | value ' + ' '.join(
                    [f'k={k+1}:{ep_value_loss_per_k[k]/ep_n:.4f}' for k in range(K)])
            print(line)

    trans_model.eval()
    return trans_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--K', type=int, default=5,
                        help='Multi-step rollout horizon for the loss')
    parser.add_argument('--n_random_eps', type=int, default=300)
    parser.add_argument('--n_policy_eps', type=int, default=300)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--value_align_coef', type=float, default=0.0,
                        help='Weight on value-alignment loss (uses trained V_eta).')
    args = parser.parse_args()

    print(f'Loading source ckpt: {args.model_path}')
    ae_model, _trans_orig, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_kind != 'continuous':
        raise NotImplementedError(
            f'Phase G first-pass implementation handles only continuous trans models; '
            f'got trans_kind={trans_kind!r}. VQ would require straight-through estimators '
            f'on the codebook lookup; deferred to a follow-up.')

    ae_model.eval()
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    hidden = getattr(margs, 'trans_hidden', 256)
    depth = getattr(margs, 'trans_depth', 3)

    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    policy = None
    if 'policy_state_dict' in src:
        try:
            policy = build_mlp(L * embedding_dim,
                               src['args'].get('policy_hidden', [256, 256]),
                               env.action_space.n,
                               src['args'].get('rl_activation', 'relu'))
            policy.load_state_dict(src['policy_state_dict'])
            policy = policy.to(args.device).eval()
            print('  policy loaded for buffer-mix')
        except Exception as e:
            print(f'  policy load failed: {e}; using random-only buffer')
            policy = None

    print(f'\n[1/3] Collecting trajectory buffer (K={args.K}, '
          f'{args.n_random_eps} random + {args.n_policy_eps} policy eps)')
    t0 = time.time()
    buf = collect_traj_buffer(env, ae_model, policy,
                              args.n_random_eps, args.n_policy_eps,
                              args.max_steps, args.K, args.device, trans_kind)
    print(f'  collected {len(buf["actions"])} K-step samples in {time.time()-t0:.1f}s')

    # Optional: load trained value head for value-alignment loss
    value_head = None
    if args.value_align_coef > 0:
        if 'critic_state_dict' not in src:
            print('  WARN: --value_align_coef > 0 but no critic_state_dict in src ckpt; disabling.')
        else:
            try:
                value_head = build_mlp(L * embedding_dim,
                                       src['args'].get('critic_hidden', [256, 256]),
                                       1,
                                       src['args'].get('rl_activation', 'relu'))
                value_head.load_state_dict(src['critic_state_dict'])
                value_head = value_head.to(args.device).eval()
                print('  value head V_eta loaded for value-alignment loss')
            except Exception as e:
                print(f'  value head load failed: {e}; disabling value-alignment')
                value_head = None

    print(f'\n[2/3] Multi-step training ({args.epochs} epochs, K={args.K}, '
          f'value_align_coef={args.value_align_coef})')
    t0 = time.time()
    trans_ms = train_multistep_continuous(
        buf, env, ae_model, embedding_dim, hidden, depth, args.K,
        args.device, args.epochs, args.batch_size, args.lr,
        value_head=value_head, value_align_coef=args.value_align_coef)
    print(f'  trained in {time.time()-t0:.1f}s')

    print(f'\n[3/3] Saving multistep WM ckpt to {args.output_path}')
    out = dict(src)
    out['trans_model_state_dict'] = {k: v.cpu() for k, v in trans_ms.state_dict().items()}
    out['args']['phaseG_K'] = args.K
    out['args']['phaseG_epochs'] = args.epochs
    out['args']['phaseG_n_samples'] = int(len(buf['actions']))
    out['args']['phaseG_source_ckpt'] = args.model_path
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(out, args.output_path)
    print('  saved.  Run analyze_wm_multistep.py + analyze_wm_planning.py to compare.')


if __name__ == '__main__':
    main()
