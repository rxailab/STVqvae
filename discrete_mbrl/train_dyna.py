"""
Full Dyna training: PPO with WM-augmented imagined rollouts.

Loads a frozen encoder + frozen world model from a source checkpoint, trains a
fresh policy + critic via PPO, and additionally augments policy/value updates
with imagined rollouts produced by the frozen WM.

Comparison conditions (set --wm_mode):
    none       : vanilla PPO baseline (no imagined rollouts)
    online     : Dyna with the original online-trained WM (paper baseline)
    oracle     : Dyna with the offline-trained oracle WM (Phase E2)
    calib      : Dyna with the head-calibrated oracle WM (Phase I)
    multistep  : Dyna with the multi-step + value-aligned WM (Phase H)
    conv       : Dyna with the spatial-conv oracle WM (caveat 3)
    bootstrap  : Phase L — use the WM only as a value-extrapolation source. We
                 do NOT add imagined samples to the PPO buffer. Instead, the
                 last-step bootstrap value is replaced by an MVE-style
                 K-step imagined return from each real terminal state:
                 V_boot = sum_k (prod γ̂) r̂_k + (prod γ̂)^K * V(ẑ_K).
                 This tests whether the WM helps even when we don't trust
                 its individual imagined transitions enough to train on them.

Headline metric: final policy reward. If WM-Dyna underperforms vanilla PPO,
the paper's negative claim hardens: even our best WMs can't help a planner.

Implementation: minimal PPO with GAE on the real-env buffer; at each update we
ALSO unroll the WM K_imagine steps from a sampled batch of real states under
the current policy's actions, compute imagined (r̂, V̂, γ̂) trajectories, GAE
the imagined trajectories, and concatenate with the real buffer for the PPO
loss. The policy gradient receives both real and imagined samples weighted by
their relative buffer sizes.

Usage:
    python train_dyna.py \\
        --src_ckpt path/to/source.pt --wm_ckpt path/to/wm.pt --wm_mode oracle \\
        --total_steps 1000000 --imagine_K 5 --imagine_ratio 0.5 \\
        --output_path path/to/dyna_ckpt.pt
"""
import argparse
import json
import os
import sys
import time
import types
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs  # noqa: E402

from analyze_wm_multistep import load_checkpoint  # noqa: E402

from analyze_wm_phaseB_e8 import (  # noqa: E402
    build_mlp, encode_flat,
)


# ── Vector env ──────────────────────────────────────────────────────────

class VectorEnv:
    """Minimal sync vector env with same API as gymnasium.SyncVectorEnv."""
    def __init__(self, env_name, num_envs):
        self.envs = [make_env(env_name) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self._obs = [None] * num_envs

    def reset(self):
        outs = [e.reset() for e in self.envs]
        obs_list = [o[0] if isinstance(o, tuple) else o for o in outs]
        self._obs = obs_list
        return np.stack(obs_list)

    def step(self, actions):
        next_obs, rewards, dones = [], [], []
        for i, a in enumerate(actions):
            sr = self.envs[i].step(int(a))
            o = sr[0] if isinstance(sr, tuple) else sr
            r = sr[1] if len(sr) >= 2 else 0.0
            d = sr[2] if len(sr) >= 3 else False
            if d:
                rs = self.envs[i].reset()
                o = rs[0] if isinstance(rs, tuple) else rs
            next_obs.append(o); rewards.append(float(r)); dones.append(bool(d))
        self._obs = next_obs
        return np.stack(next_obs), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool)


# ── GAE ─────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """rewards, values, dones: (T, N) arrays. last_value: (N,). Returns advantages, returns."""
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_v * non_terminal - values[t]
        last_adv = delta + gamma * lam * non_terminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


# ── Dyna trainer ────────────────────────────────────────────────────────

def train_dyna(args):
    device = args.device
    print(f'Loading source ckpt: {args.src_ckpt}')
    ae_model, _trans_orig, trans_kind, single_env, margs = load_checkpoint(args.src_ckpt, device)
    ae_model.eval()
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    input_dim = L * embedding_dim
    n_actions = single_env.action_space.n

    # Optional WM for Dyna augmentation
    wm_model = None
    if args.wm_mode != 'none':
        if not args.wm_ckpt:
            raise ValueError(f'--wm_mode={args.wm_mode} requires --wm_ckpt')
        print(f'Loading WM ckpt: {args.wm_ckpt}')
        _ae2, wm_model, _kind, _env2, _margs = load_checkpoint(args.wm_ckpt, device)
        if wm_model is None:
            raise RuntimeError('WM ckpt has no transition model.')
        wm_model.eval()
        for p in wm_model.parameters():
            p.requires_grad = False
        del _ae2  # use the source encoder, not the WM ckpt's

    # Fresh policy + critic
    policy = build_mlp(input_dim, [256, 256], n_actions, 'relu').to(device)
    critic = build_mlp(input_dim, [256, 256], 1, 'relu').to(device)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=args.lr)

    # Vector env
    venv = VectorEnv(margs.env_name, args.num_envs)
    obs = venv.reset()  # (N, H, W, C) or whatever

    log = {'step': [], 'mean_return_window': [], 'wm_mode': args.wm_mode}
    return_window = deque(maxlen=50)
    ep_returns = np.zeros(args.num_envs, dtype=np.float32)

    n_updates = args.total_steps // (args.n_steps_per_update * args.num_envs)
    print(f'Training {n_updates} updates ({args.total_steps} env steps total)')
    print(f'  Dyna mode: {args.wm_mode}  imagine_K={args.imagine_K}  imagine_ratio={args.imagine_ratio}')

    t0 = time.time()
    for update in range(n_updates):
        # --- Real-env rollout ---
        T = args.n_steps_per_update
        z_buf      = np.zeros((T, args.num_envs, input_dim), dtype=np.float32)
        action_buf = np.zeros((T, args.num_envs), dtype=np.int64)
        logp_buf   = np.zeros((T, args.num_envs), dtype=np.float32)
        reward_buf = np.zeros((T, args.num_envs), dtype=np.float32)
        value_buf  = np.zeros((T, args.num_envs), dtype=np.float32)
        done_buf   = np.zeros((T, args.num_envs), dtype=bool)

        for t in range(T):
            obs_t = torch.from_numpy(np.stack([preprocess_obs([o])[0] for o in obs])).float().to(device)
            with torch.no_grad():
                if trans_kind == 'discrete':
                    z = ae_model.encode(obs_t, return_quantized=True)
                else:
                    z = ae_model.encode(obs_t)
                logits = policy(z)
                v = critic(z).squeeze(-1)
                probs = F.softmax(logits, dim=-1)
                a = torch.multinomial(probs, 1).squeeze(-1)
                logp = F.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(-1)).squeeze(-1)
            z_buf[t] = z.cpu().numpy()
            action_buf[t] = a.cpu().numpy()
            logp_buf[t] = logp.cpu().numpy()
            value_buf[t] = v.cpu().numpy()

            obs, r, d = venv.step(a.cpu().numpy())
            reward_buf[t] = r
            done_buf[t] = d
            ep_returns += r
            for i in range(args.num_envs):
                if d[i]:
                    return_window.append(float(ep_returns[i]))
                    ep_returns[i] = 0.0

        # bootstrap value for last state
        with torch.no_grad():
            obs_t = torch.from_numpy(np.stack([preprocess_obs([o])[0] for o in obs])).float().to(device)
            if trans_kind == 'discrete':
                z_last = ae_model.encode(obs_t, return_quantized=True)
            else:
                z_last = ae_model.encode(obs_t)
            last_v = critic(z_last).squeeze(-1).cpu().numpy()

            # Phase L: MVE-style bootstrap — replace last_v with the WM's
            # K-step imagined return starting from the real terminal state.
            # No imagined samples are appended to the PPO buffer; the WM only
            # contributes a (better-conditioned, hopefully) value target.
            if args.wm_mode == 'bootstrap' and trans_kind != 'discrete':
                cur = z_last.view(-1, input_dim)
                G = torch.zeros(cur.shape[0], device=device)
                disc = torch.ones(cur.shape[0], device=device)
                for k in range(args.imagine_K):
                    logits_k = policy(cur)
                    probs_k = F.softmax(logits_k, dim=-1)
                    a_k = torch.multinomial(probs_k, 1).squeeze(-1)
                    z_next, r_pred, g_pred = wm_model(cur, a_k)
                    G = G + disc * r_pred.squeeze(-1)
                    disc = disc * g_pred.squeeze(-1)
                    cur = z_next.view(-1, input_dim)
                v_K = critic(cur).squeeze(-1)
                last_v_imag = (G + disc * v_K).cpu().numpy()
                # Average WM-bootstrap with critic-bootstrap (50/50). Pure WM
                # would be too aggressive given the per-step quality is finite.
                last_v = 0.5 * last_v + 0.5 * last_v_imag

        adv_real, ret_real = compute_gae(reward_buf, value_buf, done_buf, last_v,
                                         gamma=args.gamma, lam=args.gae_lambda)

        # --- Imagined rollout (Dyna augmentation) ---
        # Bootstrap mode skips the augmentation entirely; WM is used only for
        # the last-state value target above.
        z_imag, action_imag, logp_imag, value_imag, adv_imag, ret_imag = [], [], [], [], [], []
        if args.wm_mode not in ('none', 'bootstrap') and args.imagine_ratio > 0:
            n_imag_starts = int(T * args.num_envs * args.imagine_ratio)
            # Sample random starting states from the real buffer
            flat_z = z_buf.reshape(-1, input_dim)
            idx = np.random.choice(flat_z.shape[0], size=n_imag_starts, replace=True)
            z_start = torch.from_numpy(flat_z[idx]).to(device)

            cur_z = z_start
            ep_logp, ep_a, ep_z, ep_v, ep_r, ep_g = [], [], [], [], [], []
            with torch.no_grad():
                for k in range(args.imagine_K):
                    logits = policy(cur_z)
                    v = critic(cur_z).squeeze(-1)
                    probs = F.softmax(logits, dim=-1)
                    a = torch.multinomial(probs, 1).squeeze(-1)
                    logp = F.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(-1)).squeeze(-1)
                    if trans_kind == 'discrete':
                        # Need code-index input for discrete WM. Skip for simplicity.
                        # (Imagine ratio for discrete: codebook lookup of cur_z is not reliable
                        # without inverse mapping; we restrict imagined Dyna to continuous models.)
                        break
                    z_next, r_pred, g_pred = wm_model(cur_z, a)
                    ep_z.append(cur_z.cpu().numpy())
                    ep_a.append(a.cpu().numpy())
                    ep_logp.append(logp.cpu().numpy())
                    ep_v.append(v.cpu().numpy())
                    ep_r.append(r_pred.squeeze(-1).cpu().numpy())
                    ep_g.append(g_pred.squeeze(-1).cpu().numpy())
                    cur_z = z_next.view(-1, input_dim)
                # bootstrap last value
                if ep_z:
                    last_v_imag = critic(cur_z).squeeze(-1).cpu().numpy()

            if ep_z:
                z_arr   = np.stack(ep_z)        # (K, B, D)
                a_arr   = np.stack(ep_a)
                lp_arr  = np.stack(ep_logp)
                v_arr   = np.stack(ep_v)
                r_arr   = np.stack(ep_r)
                g_arr   = np.stack(ep_g)
                # GAE on imagined trajectory (use predicted gammas)
                adv = np.zeros_like(r_arr)
                last_adv = np.zeros(r_arr.shape[1], dtype=np.float32)
                for t in reversed(range(args.imagine_K)):
                    next_v = last_v_imag if t == args.imagine_K - 1 else v_arr[t + 1]
                    delta = r_arr[t] + g_arr[t] * next_v - v_arr[t]
                    last_adv = delta + g_arr[t] * args.gae_lambda * last_adv
                    adv[t] = last_adv
                ret = adv + v_arr
                z_imag.append(z_arr.reshape(-1, input_dim))
                action_imag.append(a_arr.reshape(-1))
                logp_imag.append(lp_arr.reshape(-1))
                value_imag.append(v_arr.reshape(-1))
                adv_imag.append(adv.reshape(-1))
                ret_imag.append(ret.reshape(-1))

        # --- PPO update on (real + imagined) ---
        z_all = z_buf.reshape(-1, input_dim)
        a_all = action_buf.reshape(-1)
        lp_all = logp_buf.reshape(-1)
        adv_all = adv_real.reshape(-1)
        ret_all = ret_real.reshape(-1)

        if z_imag:
            z_all = np.concatenate([z_all] + z_imag, axis=0)
            a_all = np.concatenate([a_all] + action_imag, axis=0)
            lp_all = np.concatenate([lp_all] + logp_imag, axis=0)
            adv_all = np.concatenate([adv_all] + adv_imag, axis=0)
            ret_all = np.concatenate([ret_all] + ret_imag, axis=0)

        # Normalize advantages
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        z_t   = torch.from_numpy(z_all).float().to(device)
        a_t   = torch.from_numpy(a_all).long().to(device)
        lp_t  = torch.from_numpy(lp_all).float().to(device)
        adv_t = torch.from_numpy(adv_all).float().to(device)
        ret_t = torch.from_numpy(ret_all).float().to(device)

        N = z_t.shape[0]
        for _ in range(args.ppo_iters):
            perm = torch.randperm(N)
            for s in range(0, N, args.ppo_batch_size):
                idxs = perm[s:s + args.ppo_batch_size]
                logits_b = policy(z_t[idxs])
                v_b = critic(z_t[idxs]).squeeze(-1)
                lp_b = F.log_softmax(logits_b, dim=-1).gather(1, a_t[idxs].unsqueeze(-1)).squeeze(-1)
                ratio = (lp_b - lp_t[idxs]).exp()
                surr1 = ratio * adv_t[idxs]
                surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * adv_t[idxs]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(v_b, ret_t[idxs])
                ent = -(F.softmax(logits_b, dim=-1) * F.log_softmax(logits_b, dim=-1)).sum(-1).mean()
                loss = policy_loss + 0.5 * value_loss - args.ppo_entropy_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), args.ppo_max_grad_norm)
                opt.step()

        if (update + 1) % 10 == 0 or update == 0:
            mr = float(np.mean(return_window)) if return_window else 0.0
            elapsed = time.time() - t0
            steps_done = (update + 1) * T * args.num_envs
            print(f'  update {update+1:4d}/{n_updates}  steps={steps_done:>7}  '
                  f'mean_return_window={mr:+.3f}  elapsed={elapsed:.1f}s')
            log['step'].append(steps_done)
            log['mean_return_window'].append(mr)

    # Save
    out = {
        'policy_state_dict': {k: v.cpu() for k, v in policy.state_dict().items()},
        'critic_state_dict': {k: v.cpu() for k, v in critic.state_dict().items()},
        'log': log,
        'args': vars(args),
    }
    torch.save(out, args.output_path)
    print(f'\nSaved {args.output_path}')
    print(f'Final mean return (last 50 eps): {np.mean(return_window) if return_window else 0.0:+.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt', required=True, help='Source ckpt for the encoder.')
    parser.add_argument('--wm_ckpt', default=None,
                        help='WM ckpt for Dyna augmentation; required unless --wm_mode none.')
    parser.add_argument('--wm_mode', default='none',
                        choices=['none', 'online', 'oracle', 'calib', 'multistep', 'conv', 'bootstrap'])
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--total_steps', type=int, default=1000000)
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--n_steps_per_update', type=int, default=128)
    parser.add_argument('--imagine_K', type=int, default=5)
    parser.add_argument('--imagine_ratio', type=float, default=0.5,
                        help='Imagined samples per update / real samples per update.')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--ppo_clip', type=float, default=0.2)
    parser.add_argument('--ppo_iters', type=int, default=4)
    parser.add_argument('--ppo_batch_size', type=int, default=256)
    parser.add_argument('--ppo_entropy_coef', type=float, default=0.01)
    parser.add_argument('--ppo_max_grad_norm', type=float, default=0.5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    train_dyna(args)


if __name__ == '__main__':
    main()
