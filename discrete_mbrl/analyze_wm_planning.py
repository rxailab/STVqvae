"""
Phase D-light: Imagined return vs actual return.

The WM has a built-in reward head r_hat and discount head gamma_hat. A planner
using the WM bootstraps value estimates from imagined rollouts:
    G_hat_0 = sum_{k=0..K} (prod_{j=0..k-1} gamma_hat(z_hat_j)) * r_hat(z_hat_k)
                 + (prod_{j=0..K-1} gamma_hat) * V(z_hat_K)

We compare three quantities along the same trajectory under the trained policy:
    G_real      : Sum_{t} gamma^t r_t        (actual env rewards)
    G_imagined  : Sum_{k} prod(gamma_hat) r_hat(z_hat_k) + bootstrap V(z_hat_K)
    G_critic    : V_eta(z*_0)                (what the trained critic predicts at t=0)

A planner that systematically over-estimates G via G_imagined will commit to
trajectories that don't pay out -- the "imagined-rollout reward hallucination"
the discussion section flags. We measure the over-estimate per checkpoint as:
    bias[K]      = mean(G_imagined - G_real)
    overestimate = fraction of episodes where G_imagined > G_real + 0.1
    rank_corr    = Pearson(G_imagined, G_real) across episodes

A WM-based planner is usable iff G_imagined and G_real have high rank correlation;
the planner trusts the WM to rank candidate trajectories, and miscalibrated bias
matters less than scrambled ranking.

Usage:
    python analyze_wm_planning.py --model_path ckpt.pt --output_json out.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env  # noqa: E402

from analyze_wm_multistep import load_checkpoint, encode_codes  # noqa: E402

from analyze_wm_phaseB_e8 import (  # noqa: E402
    build_mlp, encode_flat,
)


def collect_policy_episodes(env, ae_model, policy, n_episodes, max_steps,
                            device, trans_kind, stochastic=False):
    """Run the trained policy and record per-step (z_t, action_t, reward_t).
    `stochastic=True` samples actions from softmax(logits) rather than argmax;
    this matches training-time PPO behaviour and generates more diverse returns,
    which makes the imagined-vs-actual rank correlation well-defined when the
    deterministic policy would otherwise produce near-constant returns."""
    import torch.nn.functional as _F
    episodes = []
    for ep in range(n_episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        ep_z, ep_a, ep_r, ep_codes = [], [], [], []
        for t in range(max_steps):
            z = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
            with torch.no_grad():
                logits = policy(torch.from_numpy(z).unsqueeze(0).to(device))
                if stochastic:
                    probs = _F.softmax(logits.squeeze(0), dim=-1)
                    a = int(torch.multinomial(probs, 1).item())
                else:
                    a = int(logits.argmax(dim=-1).item())
            ep_z.append(z.astype(np.float32))
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
            'z':       np.stack(ep_z),
            'a':       np.array(ep_a, dtype=np.int64),
            'r':       np.array(ep_r, dtype=np.float32),
            'codes':   np.stack(ep_codes) if ep_codes else None,
        })
        if (ep + 1) % 25 == 0:
            print(f'  policy ep {ep+1}/{n_episodes}; mean return={np.mean([e["r"].sum() for e in episodes]):.3f}')
    return episodes


def imagined_return(trans_model, ae_model, value_head, ep, K, device,
                    trans_kind, gamma_default=0.99):
    """Free-run the WM K steps from the episode's start, accumulating
    imagined reward * imagined discount, then add bootstrap V(z_hat_K).
    Returns G_imagined for the rollout starting at t=0."""
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L
    actions = ep['a'][:K]
    if len(actions) == 0:
        return 0.0

    G = 0.0
    discount_acc = 1.0

    if trans_kind == 'discrete':
        cur_codes = torch.from_numpy(ep['codes'][0]).unsqueeze(0).long().to(device)
        for k in range(len(actions)):
            a_t = torch.tensor([int(actions[k])], device=device, dtype=torch.long)
            with torch.no_grad():
                logits, r_hat, g_hat = trans_model(cur_codes, a_t, return_logits=True)
            G += float(discount_acc * float(r_hat.item()))
            discount_acc *= float(g_hat.item()) if g_hat is not None else gamma_default
            # Free-run: feed argmax back as next-step input
            cur_codes = logits.argmax(dim=1)
        # Bootstrap with critic on the last predicted state
        cb = ae_model.quantizer._embedding.weight.detach()
        z_hat_K = cb[cur_codes].permute(0, 2, 1).reshape(1, L * embedding_dim)
        with torch.no_grad():
            v_K = float(value_head(z_hat_K).item())
        G += discount_acc * v_K
    else:
        cur = torch.from_numpy(ep['z'][0]).unsqueeze(0).to(device)
        for k in range(len(actions)):
            a_t = torch.tensor([int(actions[k])], device=device, dtype=torch.long)
            with torch.no_grad():
                states, r_hat, g_hat = trans_model(cur, a_t)
            G += float(discount_acc * float(r_hat.item()))
            discount_acc *= float(g_hat.item()) if g_hat is not None else gamma_default
            cur = states.view(1, -1)
        with torch.no_grad():
            v_K = float(value_head(cur).item())
        G += discount_acc * v_K
    return G


def actual_return(ep, gamma=0.99):
    """Discounted sum of rewards from t=0."""
    r = ep['r']
    discounts = gamma ** np.arange(len(r))
    return float((discounts * r).sum())


def critic_value(value_head, ep, device):
    z0 = torch.from_numpy(ep['z'][0]).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(value_head(z0).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--horizons', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--stochastic', action='store_true',
                        help='Sample actions from policy softmax (matches training).')
    args = parser.parse_args()

    print(f'Loading {args.model_path}')
    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_model is None:
        raise RuntimeError('No transition model -- cannot run planning analysis.')

    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L
    input_dim = L * embedding_dim

    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    n_actions = env.action_space.n
    policy = build_mlp(input_dim, src['args'].get('policy_hidden', [256, 256]), n_actions, 'relu')
    critic = build_mlp(input_dim, src['args'].get('critic_hidden', [256, 256]), 1, 'relu')
    policy.load_state_dict(src['policy_state_dict'])
    critic.load_state_dict(src['critic_state_dict'])
    policy = policy.to(args.device).eval()
    critic = critic.to(args.device).eval()

    print(f'\n[1/2] Collecting {args.n_episodes} policy episodes (stochastic={args.stochastic})')
    episodes = collect_policy_episodes(env, ae_model, policy, args.n_episodes,
                                       args.max_steps, args.device, trans_kind,
                                       stochastic=args.stochastic)
    ep_real = [actual_return(ep, args.gamma) for ep in episodes]
    ep_critic = [critic_value(critic, ep, args.device) for ep in episodes]
    succ_rate = float(sum(r > 0 for r in ep_real) / len(ep_real))
    # Phase K' — the CRITIC baseline. rank_corr(V_critic(z*_0), G_real) is the
    # ceiling that any WM-based bootstrap can achieve at K=1, since at K=1 the
    # imagined return is r̂ + γ̂·V(ẑ_1) ≈ r_true + γ·V(z_true_1), which is just
    # a TD-1 estimate built on top of V_critic. If this is also ≈ 0, the
    # bottleneck is the critic / task stochasticity, not the WM.
    ep_critic_arr = np.array(ep_critic, dtype=np.float32)
    ep_real_arr   = np.array(ep_real,   dtype=np.float32)
    if ep_critic_arr.std() > 1e-9 and ep_real_arr.std() > 1e-9:
        critic_rank_corr = float(np.corrcoef(ep_critic_arr, ep_real_arr)[0, 1])
    else:
        critic_rank_corr = None
    crc_str = f'{critic_rank_corr:+.3f}' if critic_rank_corr is not None else '--'
    print(f'  episodes: {len(episodes)}  succ_rate: {succ_rate:.0%}  '
          f'mean G_real: {np.mean(ep_real):+.4f}  '
          f'rank_corr(V_critic, G_real)={crc_str}')

    print(f'\n[2/2] Computing imagined returns at horizons {args.horizons}')
    per_K = {}
    for K in args.horizons:
        ep_imag = []
        for ep in episodes:
            if len(ep['a']) < 1:
                continue
            ep_imag.append(imagined_return(trans_model, ae_model, critic, ep, K,
                                           args.device, trans_kind, args.gamma))
        ep_imag = np.array(ep_imag, dtype=np.float32)
        ep_real_a = np.array(ep_real, dtype=np.float32)[:len(ep_imag)]
        bias = float((ep_imag - ep_real_a).mean())
        overest = float((ep_imag > ep_real_a + 0.1).mean())
        if ep_imag.std() > 1e-9 and ep_real_a.std() > 1e-9:
            rank_corr = float(np.corrcoef(ep_imag, ep_real_a)[0, 1])
        else:
            rank_corr = None
        per_K[K] = {
            'bias_mean':       bias,
            'overestimate_rate': overest,
            'rank_corr':       rank_corr,
            'G_imagined_mean': float(ep_imag.mean()),
            'G_imagined_std':  float(ep_imag.std()),
            'G_real_mean':     float(ep_real_a.mean()),
            'G_real_std':      float(ep_real_a.std()),
            'n':               int(len(ep_imag)),
        }

    # Console
    print('\n=== Phase D-light: imagined vs real return ===')
    print(f'  succ_rate={succ_rate:.0%}  mean V_critic(z*_0)={np.mean(ep_critic):+.4f}  mean G_real={np.mean(ep_real):+.4f}')
    print(f'  {"K":>3}  {"G_imag":>10}  {"G_real":>10}  {"bias":>10}  {"overest":>9}  {"rank_corr":>10}')
    for K in args.horizons:
        d = per_K[K]
        rc = f'{d["rank_corr"]:+.3f}' if d['rank_corr'] is not None else '   --'
        print(f'  {K:>3}  {d["G_imagined_mean"]:>+10.4f}  {d["G_real_mean"]:>+10.4f}  '
              f'{d["bias_mean"]:>+10.4f}  {d["overestimate_rate"]:>9.2%}  {rc:>10}')

    # Save
    result = {
        'model_path': args.model_path,
        'trans_model_type': trans_kind,
        'env_name': margs.env_name,
        'gamma': args.gamma,
        'n_episodes': len(episodes),
        'success_rate': succ_rate,
        'G_real_mean': float(np.mean(ep_real)),
        'V_critic_z0_mean': float(np.mean(ep_critic)),
        'V_critic_rank_corr': critic_rank_corr,
        'horizons': args.horizons,
        'per_horizon': {str(K): v for K, v in per_K.items()},
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved {args.output_json}')


if __name__ == '__main__':
    main()
