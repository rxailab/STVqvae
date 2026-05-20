"""Phase II Cell D: action-uniform oracle WM training.

Hypothesis to test (Phase II Cell C finding, 2026-05-12): policy-only buffers
collapse the WM's action-conditioning because each policy state appears with
~1 dominant action, never with the alternatives. The WM never learns
T(z, a) for a ≠ π(z).

Two ways to repair this without going back to random-action buffers (which
already work — §4.8 — but don't help on collapsed regimes where the encoder
itself is broken on random data):

  - --mode mixed_step  (F1a): at each step, take action ~ Uniform(K) with
    prob `mix_prob`, else policy(z). Simple, no env cloning. Action diversity
    is mixed but not per-state-uniform.

  - --mode branch      (F1b): at each policy state, clone the env and step
    each non-policy action separately; store all (state, a_alt, s_alt')
    transitions in the buffer. The trajectory continues under the policy
    action. This is per-state action uniformity — the cleanest test.

Matching buffer sizes to Cell C (~120k transitions):
  - mixed_step: 600 eps × 200 steps = 120k.
  - branch (with n_branches = n_actions - 1 = 6 on MiniGrid):
        ~85 eps × 200 steps × 7 actions = 120k. Use n_episodes=85.
"""
import argparse
import copy
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from train_oracle_wm import (
    load_checkpoint, build_mlp, encode_flat, encode_codes,
    TransDataset, train_oracle_discrete, train_oracle_continuous,
)


def collect_mixed_step_buffer(env, ae_model, policy, n_episodes, max_steps,
                              mix_prob, device, trans_kind):
    """F1a: per-step mix of policy and uniform actions."""
    z_in, z_out, codes_in, codes_out, actions = [], [], [], [], []
    n_actions = env.action_space.n
    for ep_idx in range(n_episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        z_t = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
        c_t = (encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
               if trans_kind == 'discrete' else None)
        for step in range(max_steps):
            if np.random.rand() < mix_prob:
                a = int(np.random.randint(n_actions))
            else:
                with torch.no_grad():
                    a = int(policy(torch.from_numpy(z_t).unsqueeze(0).to(device))
                            .argmax(dim=-1).item())
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            done = sr[2] if len(sr) >= 3 else False
            obs_t = torch.from_numpy(obs_next).float()
            z_tp1 = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
            c_tp1 = (encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
                     if trans_kind == 'discrete' else None)
            z_in.append(z_t); z_out.append(z_tp1); actions.append(a)
            if trans_kind == 'discrete':
                codes_in.append(c_t); codes_out.append(c_tp1)
            z_t = z_tp1; c_t = c_tp1
            if done: break
        if (ep_idx + 1) % 50 == 0:
            print(f'  [mixed_step] {ep_idx + 1}/{n_episodes} eps, {len(actions)} trans')
    return _to_buf(z_in, z_out, actions, codes_in, codes_out, trans_kind)


def collect_branch_buffer(env, ae_model, policy, n_episodes, max_steps,
                          device, trans_kind):
    """F1b: at each policy state, clone env and step every non-policy action.
    Per-state action-uniform buffer (only the trajectory advances under policy)."""
    z_in, z_out, codes_in, codes_out, actions = [], [], [], [], []
    n_actions = env.action_space.n

    for ep_idx in range(n_episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        z_t = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
        c_t = (encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
               if trans_kind == 'discrete' else None)

        for step in range(max_steps):
            # Policy action.
            with torch.no_grad():
                a_pi = int(policy(torch.from_numpy(z_t).unsqueeze(0).to(device))
                           .argmax(dim=-1).item())

            # Branch every non-policy action by env clone.
            for alt_a in range(n_actions):
                if alt_a == a_pi: continue
                try:
                    env_clone = copy.deepcopy(env)
                except Exception:
                    # If deepcopy is too slow / fails, fall back to skipping branches.
                    continue
                sr_alt = env_clone.step(alt_a)
                obs_alt = sr_alt[0] if isinstance(sr_alt, tuple) else sr_alt
                obs_alt_t = torch.from_numpy(obs_alt).float()
                z_alt = encode_flat(ae_model, obs_alt_t, device, trans_kind).squeeze(0).cpu().numpy()
                z_in.append(z_t); z_out.append(z_alt); actions.append(alt_a)
                if trans_kind == 'discrete':
                    c_alt = encode_codes(ae_model, obs_alt_t, device).squeeze(0).cpu().numpy()
                    codes_in.append(c_t); codes_out.append(c_alt)

            # Trajectory advances under policy action.
            sr = env.step(a_pi)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            done = sr[2] if len(sr) >= 3 else False
            obs_t = torch.from_numpy(obs_next).float()
            z_tp1 = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
            c_tp1 = (encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
                     if trans_kind == 'discrete' else None)
            z_in.append(z_t); z_out.append(z_tp1); actions.append(a_pi)
            if trans_kind == 'discrete':
                codes_in.append(c_t); codes_out.append(c_tp1)
            z_t = z_tp1; c_t = c_tp1
            if done: break

        if (ep_idx + 1) % 10 == 0:
            print(f'  [branch] {ep_idx + 1}/{n_episodes} eps, {len(actions)} trans')
    return _to_buf(z_in, z_out, actions, codes_in, codes_out, trans_kind)


def _to_buf(z_in, z_out, actions, codes_in, codes_out, trans_kind):
    buf = {
        'z_in':    np.stack(z_in).astype(np.float32),
        'z_out':   np.stack(z_out).astype(np.float32),
        'actions': np.array(actions, dtype=np.int64),
    }
    if trans_kind == 'discrete':
        buf['codes_in']  = np.stack(codes_in).astype(np.int64)
        buf['codes_out'] = np.stack(codes_out).astype(np.int64)
    print(f'  buffer total: {len(actions)} transitions')
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--mode', choices=['mixed_step', 'branch'], required=True)
    parser.add_argument('--mix_prob', type=float, default=0.5,
                        help='For mixed_step mode: prob of random action per step.')
    parser.add_argument('--n_episodes', type=int, default=None,
                        help='Total episodes to collect. Defaults: 600 (mixed_step), 85 (branch).')
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--trans_hidden', type=int, default=None)
    parser.add_argument('--trans_depth', type=int, default=None)
    args = parser.parse_args()

    if args.n_episodes is None:
        args.n_episodes = 600 if args.mode == 'mixed_step' else 85

    print(f'Loading source ckpt: {args.model_path}')
    ae_model, _trans_orig, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    ae_model.eval()
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    n_codes = getattr(margs, 'codebook_size', 64)
    hidden = args.trans_hidden if args.trans_hidden is not None else getattr(margs, 'trans_hidden', 256)
    depth  = args.trans_depth  if args.trans_depth  is not None else getattr(margs, 'trans_depth', 3)
    print(f'  trans_model arch: hidden={hidden}, depth={depth}')

    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    n_actions = env.action_space.n
    if 'policy_state_dict' not in src:
        raise RuntimeError('Source ckpt has no policy — cannot run action-uniform training.')
    policy_hidden = src['args'].get('policy_hidden', [256, 256])
    policy = build_mlp(L * embedding_dim, policy_hidden, n_actions, 'relu')
    policy.load_state_dict(src['policy_state_dict'])
    policy = policy.to(args.device).eval()
    print(f'  policy loaded (input={L*embedding_dim}, n_actions={n_actions})')

    print(f'\n[1/3] Collecting buffer (mode={args.mode}, n_episodes={args.n_episodes})')
    t0 = time.time()
    if args.mode == 'mixed_step':
        buf = collect_mixed_step_buffer(env, ae_model, policy, args.n_episodes,
                                        args.max_steps, args.mix_prob,
                                        args.device, trans_kind)
    else:
        buf = collect_branch_buffer(env, ae_model, policy, args.n_episodes,
                                    args.max_steps, args.device, trans_kind)
    print(f'  collected in {time.time()-t0:.1f}s')

    # Diagnostic: action histogram
    action_counts = np.bincount(buf['actions'], minlength=n_actions)
    print(f'  action histogram: {action_counts.tolist()}  (uniform → {len(buf["actions"]) // n_actions})')

    print(f'\n[2/3] Training oracle WM ({args.epochs} epochs)')
    t0 = time.time()
    if trans_kind == 'discrete':
        trans_model = train_oracle_discrete(buf, env, ae_model, n_codes, embedding_dim,
                                            hidden, depth, args.device, args.epochs,
                                            args.batch_size, args.lr)
    else:
        trans_model = train_oracle_continuous(buf, env, ae_model, embedding_dim,
                                              hidden, depth, args.device, args.epochs,
                                              args.batch_size, args.lr)
    print(f'  trained in {time.time()-t0:.1f}s')

    print(f'\n[3/3] Saving to {args.output_path}')
    new_src = dict(src)
    new_src['trans_state_dict'] = trans_model.state_dict()
    new_src['args']['trans_hidden'] = hidden
    new_src['args']['trans_depth'] = depth
    new_src['oracle_meta'] = {
        'mode': args.mode,
        'mix_prob': args.mix_prob if args.mode == 'mixed_step' else None,
        'n_episodes': args.n_episodes,
        'epochs': args.epochs,
        'n_transitions': int(len(buf['actions'])),
        'action_histogram': action_counts.tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(new_src, args.output_path)
    print('  saved.')


if __name__ == '__main__':
    main()
