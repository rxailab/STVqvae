"""
Phase B (E8): Policy / value coherence between z* and ẑ_k.

Loads the trained PPO policy and value head from the checkpoint, then asks:
when the world model's predicted ẑ_k is fed into the same policy/critic that
were trained on the encoder's true z*_k, do they produce the same answers?

Metrics per horizon k:
  action_match  : argmax pi(ẑ_k) == argmax pi(z*_k)        (top-1 agreement)
  kl_div        : KL(pi(ẑ_k) || pi(z*_k))                  (distributional)
  value_mse     : MSE between V(ẑ_k) and V(z*_k)
  value_corr    : Pearson r between V(ẑ_k) and V(z*_k) across rollouts

Rollouts are collected under the trained policy itself (not random), so the
state distribution and reward signal are non-trivial. The same action sequence
the policy chose in the env is replayed through the WM to produce ẑ_k.

Usage:
    python analyze_wm_phaseB_e8.py --model_path ckpt.pt --output_json out.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env  # noqa: E402

from analyze_wm_multistep import (  # noqa: E402
    load_checkpoint,
    rollout_discrete, rollout_continuous,
)


# ── Policy / critic reconstruction ─────────────────────────────────────────

def build_mlp(input_dim, hidden_sizes, output_dim, activation='relu'):
    """Reconstruct the PPO actor / critic Sequential. State-dict keys are 1,3,5
    because index 0 is a no-op (Identity / Flatten) without parameters."""
    act_cls = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}[activation]
    layers = [nn.Identity()]
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def load_policy_critic(ckpt_path, ae_model, env, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_d = ckpt['args']
    hidden = args_d.get('policy_hidden', [256, 256])
    act = args_d.get('rl_activation', 'relu')

    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L
    input_dim = L * embedding_dim

    n_actions = env.action_space.n

    policy = build_mlp(input_dim, hidden, n_actions, activation=act)
    critic = build_mlp(input_dim, args_d.get('critic_hidden', hidden), 1, activation=act)
    policy.load_state_dict(ckpt['policy_state_dict'])
    critic.load_state_dict(ckpt['critic_state_dict'])
    policy = policy.to(device).eval()
    critic = critic.to(device).eval()
    return policy, critic, n_actions, input_dim, L, embedding_dim


# ── Rollout helpers (policy-driven, not random) ────────────────────────────

def encode_flat(ae_model, obs_t, device, trans_kind):
    """Returns (1, L*D) continuous flat embedding."""
    obs_in = obs_t.unsqueeze(0).to(device)
    with torch.no_grad():
        if trans_kind == 'discrete':
            return ae_model.encode(obs_in, return_quantized=True)
        return ae_model.encode(obs_in)


def encode_codes(ae_model, obs_t, device):
    """Returns (1, L) long code indices (VQ only)."""
    obs_in = obs_t.unsqueeze(0).to(device)
    with torch.no_grad():
        codes = ae_model.encode(obs_in, return_quantized=False)
        if codes.dim() == 3:
            codes = codes.view(1, -1)
    return codes.long()


def collect_policy_rollouts(env, ae_model, policy, n_episodes, max_steps, device, trans_kind):
    """Roll out the trained policy and record z*, codes (VQ), action, reward at every step."""
    episodes = []
    for ep in range(n_episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        ep_z, ep_codes, ep_a, ep_r = [], [], [], []
        for t in range(max_steps):
            z_flat = encode_flat(ae_model, obs_t, device, trans_kind)             # (1, L*D)
            with torch.no_grad():
                logits = policy(z_flat)
                a = int(logits.argmax(dim=-1).item())
            ep_z.append(z_flat.squeeze(0).cpu().numpy().astype(np.float32))
            ep_a.append(a)
            if trans_kind == 'discrete':
                c = encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
                ep_codes.append(c)
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            r = sr[1] if len(sr) >= 2 else 0.0
            done = sr[2] if len(sr) >= 3 else False
            ep_r.append(float(r))
            obs_t = torch.from_numpy(obs_next).float()
            if done:
                break
        episodes.append({
            'z': np.stack(ep_z),
            'codes': (np.stack(ep_codes) if ep_codes else None),
            'a': np.array(ep_a, dtype=np.int64),
            'r': np.array(ep_r, dtype=np.float32),
        })
        if (ep + 1) % 10 == 0:
            print(f'  policy eps: {ep + 1}/{n_episodes}  total_steps={sum(len(e["z"]) for e in episodes)}')
    return episodes


# ── WM free-run from each starting state ──────────────────────────────────

def codes_to_flat_channel_major(codes_L, codebook_KD):
    """codes_L: (L,) long. codebook: (K, D). Returns (L*D,) channel-major flat
    matching the encoder's .reshape(B, -1) on a (B, D, H, W) tensor."""
    emb = codebook_KD[codes_L]               # (L, D)
    return emb.permute(1, 0).reshape(-1)     # (D, L) -> (D*L,)


def freerun_pairs(trans_model, ae_model, episodes, k_max, device, trans_kind):
    """For each episode, for each start t such that t + k_max <= len, run the WM
    forward k_max steps using the episode's actions. Returns lists of arrays:
        z_star[h]  : (N, D*L)
        z_hat[h]   : (N, D*L)
        r_at_h[h]  : (N,)
    where h indexes horizon 1..k_max.
    """
    z_star = {k: [] for k in range(1, k_max + 1)}
    z_hat = {k: [] for k in range(1, k_max + 1)}
    r_at = {k: [] for k in range(1, k_max + 1)}

    if trans_kind == 'discrete':
        codebook = ae_model.quantizer._embedding.weight.detach().cpu()  # (K, D)
    else:
        codebook = None

    for ep in episodes:
        T = len(ep['z'])
        for t in range(T - 1):
            steps_remaining = T - t - 1
            steps = min(k_max, steps_remaining)
            actions = ep['a'][t : t + steps].tolist()

            if trans_kind == 'discrete':
                codes_0 = torch.from_numpy(ep['codes'][t])           # (L,)
                preds = rollout_discrete(trans_model, codes_0, actions, device)  # list of (L,) long
                pred_flats = [
                    codes_to_flat_channel_major(p, codebook).numpy().astype(np.float32)
                    for p in preds
                ]
            else:
                z0 = torch.from_numpy(ep['z'][t])                    # (D*L,)
                preds = rollout_continuous(trans_model, ae_model, z0, actions, device)
                pred_flats = [p.cpu().numpy().astype(np.float32) for p in preds]

            for k in range(1, steps + 1):
                z_star[k].append(ep['z'][t + k])
                z_hat[k].append(pred_flats[k - 1])
                r_at[k].append(ep['r'][t + k - 1])  # reward at step t+k

    out = {}
    for k in range(1, k_max + 1):
        if not z_star[k]:
            continue
        out[k] = {
            'z_star': np.stack(z_star[k]),
            'z_hat':  np.stack(z_hat[k]),
            'r':      np.array(r_at[k], dtype=np.float32),
        }
    return out


# ── Coherence metrics ──────────────────────────────────────────────────────

def coherence_metrics(policy, critic, z_star_NB, z_hat_NB, device, batch=512):
    """Apply policy/critic to z*_k and ẑ_k. Returns dict with action_match, kl,
    value_mse, value_corr, plus diagnostic value distributions."""
    N = z_star_NB.shape[0]
    a_true = np.zeros(N, dtype=np.int64)
    a_wm = np.zeros(N, dtype=np.int64)
    kls = np.zeros(N, dtype=np.float32)
    v_true = np.zeros(N, dtype=np.float32)
    v_wm = np.zeros(N, dtype=np.float32)

    for start in range(0, N, batch):
        end = min(start + batch, N)
        zs = torch.from_numpy(z_star_NB[start:end]).to(device)
        zh = torch.from_numpy(z_hat_NB[start:end]).to(device)
        with torch.no_grad():
            logits_t = policy(zs)
            logits_w = policy(zh)
            v_t = critic(zs).squeeze(-1)
            v_w = critic(zh).squeeze(-1)
            log_p_t = F.log_softmax(logits_t, dim=-1)
            log_p_w = F.log_softmax(logits_w, dim=-1)
            p_w = log_p_w.exp()
            kl = (p_w * (log_p_w - log_p_t)).sum(dim=-1)
        a_true[start:end] = logits_t.argmax(dim=-1).cpu().numpy()
        a_wm[start:end]   = logits_w.argmax(dim=-1).cpu().numpy()
        kls[start:end]    = kl.cpu().numpy()
        v_true[start:end] = v_t.cpu().numpy()
        v_wm[start:end]   = v_w.cpu().numpy()

    # action match
    action_match = float((a_true == a_wm).mean())
    # KL
    kl_mean = float(kls.mean())
    kl_median = float(np.median(kls))
    # value
    value_mse = float(((v_wm - v_true) ** 2).mean())
    if v_true.std() > 1e-9 and v_wm.std() > 1e-9:
        value_corr = float(np.corrcoef(v_true, v_wm)[0, 1])
    else:
        value_corr = None
    return {
        'n': N,
        'action_match':   action_match,
        'kl_mean':        kl_mean,
        'kl_median':      kl_median,
        'value_mse':      value_mse,
        'value_corr':     value_corr,
        'v_true_mean':    float(v_true.mean()),
        'v_true_std':     float(v_true.std()),
        'v_wm_mean':      float(v_wm.mean()),
        'v_wm_std':       float(v_wm.std()),
        'a_true_dist':    {int(a): int((a_true == a).sum()) for a in np.unique(a_true)},
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()

    print(f'Loading {args.model_path}')
    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_model is None:
        raise RuntimeError('No transition model in checkpoint -- cannot run E8.')

    policy, critic, n_actions, input_dim, L, embedding_dim = load_policy_critic(
        args.model_path, ae_model, env, args.device)
    print(f'Policy/critic loaded: input_dim={input_dim}, n_actions={n_actions}')

    # 1. Policy rollouts
    print(f'\nCollecting {args.n_episodes} policy rollouts (max {args.max_steps} steps each)')
    episodes = collect_policy_rollouts(
        env, ae_model, policy, args.n_episodes, args.max_steps, args.device, trans_kind)

    # Diagnostics
    ep_lens = [len(e['z']) for e in episodes]
    ep_returns = [float(e['r'].sum()) for e in episodes]
    print(f'  episode lengths : mean={np.mean(ep_lens):.1f}  min={min(ep_lens)}  max={max(ep_lens)}')
    print(f'  episode returns : mean={np.mean(ep_returns):.3f}  successes={sum(r > 0 for r in ep_returns)}/{len(ep_returns)}')

    # 2. Free-running WM from each starting state
    k_max = max(args.horizons)
    print(f'\nFree-running WM up to k_max={k_max}')
    pairs = freerun_pairs(trans_model, ae_model, episodes, k_max, args.device, trans_kind)
    for k in args.horizons:
        n = pairs.get(k, {}).get('z_star', np.zeros((0,))).shape[0] if k in pairs else 0
        print(f'  k={k}: {n} (start, horizon) pairs')

    # 3. Coherence metrics per horizon
    print('\nApplying policy + critic to (z*_k, ẑ_k) pairs')
    per_horizon = {}
    for k in args.horizons:
        if k not in pairs:
            continue
        m = coherence_metrics(policy, critic, pairs[k]['z_star'], pairs[k]['z_hat'], args.device)
        m['reward_mean_at_k'] = float(pairs[k]['r'].mean())
        per_horizon[k] = m

    # Console
    print('\n=== E8: policy/value coherence ===')
    print(f'  {"k":>3}  {"N":>6}  {"a_match":>8}  {"kl_med":>8}  {"v_mse":>9}  {"v_corr":>8}  reward@k')
    for k in args.horizons:
        if k not in per_horizon:
            continue
        m = per_horizon[k]
        vc = f'{m["value_corr"]:+.3f}' if m['value_corr'] is not None else '   --'
        print(f'  {k:>3}  {m["n"]:>6}  {m["action_match"]:>8.3f}  {m["kl_median"]:>8.4f}  '
              f'{m["value_mse"]:>9.5f}  {vc:>8}  {m["reward_mean_at_k"]:>+.4f}')

    # 4. Save
    result = {
        'model_path': args.model_path,
        'trans_model_type': trans_kind,
        'env_name': margs.env_name,
        'horizons': args.horizons,
        'n_episodes': args.n_episodes,
        'episode_length_mean': float(np.mean(ep_lens)),
        'episode_return_mean': float(np.mean(ep_returns)),
        'episode_success_rate': float(sum(r > 0 for r in ep_returns) / len(ep_returns)),
        'per_horizon': {str(k): v for k, v in per_horizon.items()},
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved {args.output_json}')


if __name__ == '__main__':
    main()
