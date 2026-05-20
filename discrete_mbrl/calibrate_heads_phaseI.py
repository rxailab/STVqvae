"""
Phase I: head calibration on the WM's imagined-rollout distribution.

Hypothesis: planning incoherence (Phase D, Phase F-light) is at least partly
because the WM's reward and discount heads were trained on the encoder's true
state distribution z*_t but are queried at planning time on the WM's predicted
ẑ_k, which drifts out-of-distribution as k grows. The state head can be
accurate (Phase E2 oracle WM) while the reward and discount heads systematically
mispredict on imagined states.

Procedure:
  1. Load a source ckpt (online or oracle) and freeze ae_model + state_head +
     shared_layers + embeddings.
  2. Run the trained policy in env to collect (z*_t, a_t, r_t) trajectories.
  3. For each rollout, free-run the WM K steps to get imagined ẑ_1..ẑ_K, paired
     with the real action sequence and the real per-step rewards from the env.
  4. Fine-tune ONLY the reward and gamma heads on (ẑ_k, a_k) -> (r_k_real,
     γ_k_real). Everything else is frozen.
  5. Save a "calibrated" ckpt.

Re-running analyze_wm_planning.py on the calibrated ckpt then tells us whether
head calibration alone closes the rank_corr gap that survived Phase E2.

Usage:
    python calibrate_heads_phaseI.py \\
        --model_path src.pt --output_path calib.pt \\
        --n_episodes 200 --max_steps 200 --K 10 --epochs 50
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

from analyze_wm_multistep import (  # noqa: E402
    load_checkpoint, encode_codes,
)
from analyze_wm_phaseB_e8 import (  # noqa: E402
    build_mlp, encode_flat,
)


# ── Imagined-rollout buffer collection ──────────────────────────────────

def collect_imagined_buffer(env, ae_model, trans_model, policy,
                            n_episodes, max_steps, K, device, trans_kind,
                            gamma_default=0.99):
    """Roll the trained policy in the env. For each starting state, run the WM
    forward K steps using the policy's action sequence; pair each imagined ẑ_k
    with the action a_k and the *real* env reward r_k that occurred at that
    step. Returns one big buffer of (z_hat, action, reward, gamma_target) rows."""

    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L

    z_hat_buf, code_hat_buf, action_buf, reward_buf, gamma_buf = [], [], [], [], []

    for ep in range(n_episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()

        # Roll policy, recording per-step state and reward
        ep_states_long = []      # codes for VQ; None for VAE
        ep_states_flat = []      # flat continuous embedding
        ep_actions, ep_rewards, ep_dones = [], [], []
        for t in range(max_steps):
            z_flat = encode_flat(ae_model, obs_t, device, trans_kind)  # (1, L*D)
            with torch.no_grad():
                a = int(policy(z_flat).argmax(dim=-1).item())
            ep_states_flat.append(z_flat.squeeze(0).cpu().numpy().astype(np.float32))
            if trans_kind == 'discrete':
                c = encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
                ep_states_long.append(c)
            ep_actions.append(a)
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            r = sr[1] if len(sr) >= 2 else 0.0
            done = sr[2] if len(sr) >= 3 else False
            ep_rewards.append(float(r))
            ep_dones.append(bool(done))
            obs_t = torch.from_numpy(obs_next).float()
            if done:
                break

        T = len(ep_actions)
        if T < 2:
            continue

        # For each starting position t in this episode, run WM forward K steps
        # using the actual action sequence and pair imagined ẑ_k with real r_k.
        for t in range(T - 1):
            steps = min(K, T - t - 1)
            if steps == 0:
                continue
            actions = ep_actions[t:t + steps]

            if trans_kind == 'discrete':
                cur_codes = torch.from_numpy(ep_states_long[t]).unsqueeze(0).long().to(device)
                for k in range(steps):
                    a_t = torch.tensor([int(actions[k])], device=device, dtype=torch.long)
                    with torch.no_grad():
                        logits, _, _ = trans_model(cur_codes, a_t, return_logits=True)
                    pred_codes = logits.argmax(dim=1)  # (1, L)
                    # Bookkeeping: this prediction corresponds to the env-step at time t+k+1.
                    # The reward we want the head to predict is ep_rewards[t+k] (the reward
                    # received after taking action a_k from state at time t+k).
                    code_hat_buf.append(pred_codes.squeeze(0).cpu().numpy())
                    # Action used FROM the predicted state (next action in the sequence)
                    next_action = ep_actions[t + k + 1] if (t + k + 1) < T else 0
                    action_buf.append(int(next_action))
                    reward_buf.append(float(ep_rewards[t + k]))
                    # gamma target: 0 if env terminated at step t+k, else gamma_default
                    g_t = 0.0 if ep_dones[t + k] else gamma_default
                    gamma_buf.append(g_t)
                    cur_codes = pred_codes
            else:
                z = torch.from_numpy(ep_states_flat[t]).unsqueeze(0).to(device)
                for k in range(steps):
                    a_t = torch.tensor([int(actions[k])], device=device, dtype=torch.long)
                    with torch.no_grad():
                        states, _, _ = trans_model(z, a_t)
                    z_next = states.view(1, -1)
                    z_hat_buf.append(z_next.squeeze(0).cpu().numpy().astype(np.float32))
                    next_action = ep_actions[t + k + 1] if (t + k + 1) < T else 0
                    action_buf.append(int(next_action))
                    reward_buf.append(float(ep_rewards[t + k]))
                    g_t = 0.0 if ep_dones[t + k] else gamma_default
                    gamma_buf.append(g_t)
                    z = z_next

        if (ep + 1) % 25 == 0:
            n_so_far = len(action_buf)
            print(f'  ep {ep+1}/{n_episodes}: {n_so_far} (ẑ, a, r, γ) samples')

    buf = {
        'codes':   np.stack(code_hat_buf) if trans_kind == 'discrete' else None,
        'z':       np.stack(z_hat_buf)    if trans_kind != 'discrete' else None,
        'actions': np.array(action_buf, dtype=np.int64),
        'rewards': np.array(reward_buf, dtype=np.float32),
        'gammas':  np.array(gamma_buf,  dtype=np.float32),
    }
    n = len(action_buf)
    print(f'  buffer total: {n} samples, '
          f'mean reward={buf["rewards"].mean():.4f}, '
          f'frac terminal={(buf["gammas"] == 0).mean():.3f}')
    return buf


# ── Head fine-tuning ────────────────────────────────────────────────────

class CalibBuf(Dataset):
    def __init__(self, buf, kind):
        self.kind = kind
        self.actions = buf['actions']
        self.rewards = buf['rewards']
        self.gammas = buf['gammas']
        if kind == 'discrete':
            self.codes = buf['codes']
        else:
            self.z = buf['z']

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        if self.kind == 'discrete':
            return self.codes[i], self.actions[i], self.rewards[i], self.gammas[i]
        return self.z[i], self.actions[i], self.rewards[i], self.gammas[i]


def finetune_heads(trans_model, buf, trans_kind, device, epochs, batch_size, lr):
    """Freeze everything except reward_head and gamma_head; fine-tune them on
    the imagined-rollout buffer."""
    for p in trans_model.parameters():
        p.requires_grad = False
    for p in trans_model.reward_head.parameters():
        p.requires_grad = True
    for p in trans_model.gamma_head.parameters():
        p.requires_grad = True
    head_params = [p for p in trans_model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(head_params, lr=lr)
    print(f'  fine-tuning {sum(p.numel() for p in head_params):,} head params '
          f'(ckpt has {sum(p.numel() for p in trans_model.parameters()):,} total)')

    ds = CalibBuf(buf, trans_kind)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    trans_model.eval()  # keep BN/dropout in eval mode (we're not actually using them but be safe)
    for ep in range(epochs):
        loss_sum_r, loss_sum_g, n = 0.0, 0.0, 0
        for batch in loader:
            if trans_kind == 'discrete':
                codes, acts, r_t, g_t = batch
                codes = codes.long().to(device)   # (B, L)
                acts  = acts.long().to(device)
                r_t = r_t.float().to(device).unsqueeze(-1)
                g_t = g_t.float().to(device).unsqueeze(-1)
                # Forward through frozen embedding + shared layers
                with torch.no_grad():
                    embeds = trans_model.embeddings(codes)
                    flat_embeds = embeds.view(embeds.shape[0], -1)
                    acts_proc = trans_model.prepare_acts(acts)
                    input_embeds = torch.cat([flat_embeds, acts_proc], dim=1)
                    z = trans_model.shared_layers(input_embeds)
            else:
                z_in, acts, r_t, g_t = batch
                z_in = z_in.float().to(device)
                acts = acts.long().to(device)
                r_t = r_t.float().to(device).unsqueeze(-1)
                g_t = g_t.float().to(device).unsqueeze(-1)
                with torch.no_grad():
                    z_view = z_in.view(z_in.shape[0], trans_model.input_dim)
                    acts_proc = trans_model.prepare_acts(acts)
                    input_embeds = torch.cat([z_view, acts_proc], dim=1)
                    z = trans_model.shared_layers(input_embeds)

            # Trainable: heads only
            r_pred = trans_model.reward_head(z)
            g_pred_raw = trans_model.gamma_head(z)
            # The discrete model applies sigmoid to gamma in forward; the continuous
            # model returns gamma raw and then sigmoids. We always apply sigmoid here
            # for a [0,1] target.
            g_pred = torch.sigmoid(g_pred_raw)
            loss_r = F.mse_loss(r_pred, r_t)
            loss_g = F.mse_loss(g_pred, g_t)
            loss = loss_r + loss_g

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum_r += float(loss_r.item()) * codes.shape[0] if trans_kind == 'discrete' else float(loss_r.item()) * z_in.shape[0]
            loss_sum_g += float(loss_g.item()) * codes.shape[0] if trans_kind == 'discrete' else float(loss_g.item()) * z_in.shape[0]
            n += (codes.shape[0] if trans_kind == 'discrete' else z_in.shape[0])

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  ep {ep+1:3d}/{epochs}  mse_r={loss_sum_r/n:.5f}  mse_g={loss_sum_g/n:.5f}')


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--n_episodes', type=int, default=150)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--K', type=int, default=10,
                        help='Imagined-rollout horizon used to fill the buffer')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f'Loading source ckpt: {args.model_path}')
    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_model is None:
        raise RuntimeError('No transition model in checkpoint.')

    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L

    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    n_actions = env.action_space.n
    policy = build_mlp(L * embedding_dim,
                       src['args'].get('policy_hidden', [256, 256]),
                       n_actions,
                       src['args'].get('rl_activation', 'relu'))
    policy.load_state_dict(src['policy_state_dict'])
    policy = policy.to(args.device).eval()

    print(f'\n[1/3] Collecting imagined-rollout buffer ({args.n_episodes} eps, K={args.K})')
    t0 = time.time()
    buf = collect_imagined_buffer(env, ae_model, trans_model, policy,
                                  args.n_episodes, args.max_steps,
                                  args.K, args.device, trans_kind,
                                  gamma_default=args.gamma)
    print(f'  collected in {time.time()-t0:.1f}s')

    print(f'\n[2/3] Fine-tuning reward + gamma heads ({args.epochs} epochs)')
    t0 = time.time()
    trans_model.train()  # in case any layer cares; gradient is gated by requires_grad
    finetune_heads(trans_model, buf, trans_kind, args.device,
                   args.epochs, args.batch_size, args.lr)
    print(f'  trained in {time.time()-t0:.1f}s')

    print(f'\n[3/3] Saving calibrated ckpt to {args.output_path}')
    out = dict(src)
    out['trans_model_state_dict'] = {k: v.cpu() for k, v in trans_model.state_dict().items()}
    out['args']['phaseI_calib_episodes'] = args.n_episodes
    out['args']['phaseI_calib_epochs'] = args.epochs
    out['args']['phaseI_buffer_K'] = args.K
    out['args']['phaseI_source_ckpt'] = args.model_path
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(out, args.output_path)
    print('  saved.  Re-run analyze_wm_planning.py on it to test the planning fix.')


if __name__ == '__main__':
    main()
