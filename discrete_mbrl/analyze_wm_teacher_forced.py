"""
Phase E1a: Teacher-forced k-step WM accuracy.

For each step k in a rollout, compute the WM's prediction by feeding it the
*true* state z*_{k-1} (not its own prior prediction). This isolates per-step
prediction capacity from compounding error.

Free-running k-step (paper §3.3): ẑ_k = WM(WM(WM(...WM(z*_0, a_0)..., a_{k-2}), a_{k-1}))
Teacher-forced k-step (this file): ẑ_k^TF = WM(z*_{k-1}, a_{k-1})

If TF k=10 on door is also at the floor, the WM cannot do this prediction
even given a perfect input — the per-step capacity is the bottleneck and
"the WM is just a weak MLP" cannot explain the dissociation.
If TF k=10 on door is much higher than free-running k=10, the failure is
compounding error rather than per-step incapacity.

Usage:
    python analyze_wm_teacher_forced.py --model_path ckpt.pt --output_json out.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, OBJECT_TO_IDX  # noqa: E402

from analyze_wm_multistep import (  # noqa: E402
    load_checkpoint, semantic_grid, encode_codes, encode_latent,
    OBJECT_NAMES, N_CLASSES,
)


def _bucket_per_class(correct_NL, sem0_NL):
    correct = correct_NL.astype(np.float32).reshape(-1)
    classes = sem0_NL.reshape(-1)
    out = {}
    for c in range(N_CLASSES):
        m = classes == c
        n = int(m.sum())
        if n < 10:
            continue
        out[int(c)] = {
            'acc': float(correct[m].mean()),
            'count': n,
            'name': OBJECT_NAMES[int(c)],
        }
    return out


def collect_rollouts(env, ae_model, n_rollouts, k_max, device, trans_kind, n_lat_side):
    """Roll out random actions; record full per-step trajectories (codes for VQ,
    embeddings for VAE, and semantic class at every step)."""
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L

    episodes = []
    for r in range(n_rollouts):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        codes_seq, lat_seq, sem_seq, actions = [], [], [], []

        # initial step
        sem_seq.append(semantic_grid(env, n_lat_side).numpy())
        if trans_kind == 'discrete':
            codes_seq.append(encode_codes(ae_model, obs_t, device).numpy())
        else:
            lat_seq.append(encode_latent(ae_model, obs_t, device).cpu().numpy())

        for step in range(k_max):
            a = env.action_space.sample()
            actions.append(a)
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            done = sr[2] if len(sr) >= 3 else False
            obs_t = torch.from_numpy(obs_next).float()
            sem_seq.append(semantic_grid(env, n_lat_side).numpy())
            if trans_kind == 'discrete':
                codes_seq.append(encode_codes(ae_model, obs_t, device).numpy())
            else:
                lat_seq.append(encode_latent(ae_model, obs_t, device).cpu().numpy())
            if done:
                break

        episodes.append({
            'codes':   np.stack(codes_seq) if codes_seq else None,   # (T+1, L) long, VQ only
            'lat':     np.stack(lat_seq)   if lat_seq   else None,   # (T+1, D*L) float, VAE only
            'sem':     np.stack(sem_seq),                             # (T+1, L)
            'actions': np.array(actions, dtype=np.int64),             # (T,)
        })

        if (r + 1) % 50 == 0:
            print(f'  rollouts: {r + 1}/{n_rollouts}')

    return episodes


def teacher_forced_score(trans_model, ae_model, episodes, k_max, device, trans_kind):
    """For each k, compute WM(z*_{k-1}, a_{k-1}) and compare to z*_k.
    Bucket by class at t=0 (paper §3.3a)."""
    out = {}
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L

    for k in range(1, k_max + 1):
        sem0_buf = []
        correct_buf = []
        for ep in episodes:
            T = len(ep['actions'])
            if k > T:
                continue
            # input: z*_{k-1}, a_{k-1}
            if trans_kind == 'discrete':
                in_codes = torch.from_numpy(ep['codes'][k - 1]).unsqueeze(0).long().to(device)
                a_in = torch.tensor([int(ep['actions'][k - 1])], device=device, dtype=torch.long)
                with torch.no_grad():
                    out_logits, _, _ = trans_model(in_codes, a_in, return_logits=True)
                pred = out_logits.argmax(dim=1).squeeze(0).cpu().numpy()      # (L,)
                truth = ep['codes'][k]
                correct = (pred == truth).astype(np.float32)
            else:
                in_lat = torch.from_numpy(ep['lat'][k - 1]).unsqueeze(0).to(device)
                a_in = torch.tensor([int(ep['actions'][k - 1])], device=device, dtype=torch.long)
                with torch.no_grad():
                    states, _, _ = trans_model(in_lat, a_in)
                pred_LD = states.squeeze(0).view(L, embedding_dim).cpu().numpy()
                truth_LD = ep['lat'][k].reshape(L, embedding_dim)
                # cosine-argmax-within-frame
                p_n = pred_LD / (np.linalg.norm(pred_LD, axis=-1, keepdims=True) + 1e-9)
                t_n = truth_LD / (np.linalg.norm(truth_LD, axis=-1, keepdims=True) + 1e-9)
                sims = p_n @ t_n.T
                correct = (sims.argmax(axis=-1) == np.arange(L)).astype(np.float32)
            sem0_buf.append(ep['sem'][0])
            correct_buf.append(correct)

        if not sem0_buf:
            continue
        sem0_arr    = np.stack(sem0_buf)        # (N, L)
        correct_arr = np.stack(correct_buf)
        out[k] = _bucket_per_class(correct_arr, sem0_arr)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_rollouts', type=int, default=300)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()

    print(f'Loading {args.model_path}')
    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_model is None:
        raise RuntimeError('No transition model in checkpoint -- cannot run teacher-forced analysis.')

    L = ae_model.n_latent_embeds
    n_lat_side = int(round(L ** 0.5))
    k_max = max(args.horizons)

    print(f'Collecting {args.n_rollouts} rollouts up to k_max={k_max}')
    episodes = collect_rollouts(env, ae_model, args.n_rollouts, k_max,
                                args.device, trans_kind, n_lat_side)

    print('Scoring teacher-forced predictions per horizon')
    tf_per_k = teacher_forced_score(trans_model, ae_model, episodes, k_max,
                                    args.device, trans_kind)

    print('\n=== Teacher-forced WM accuracy per class ===')
    cls_order = ['wall', 'door', 'key', 'goal', 'agent']
    print(f'  {"k":>3}  ' + '  '.join(f'{n:>7}' for n in cls_order))
    for k in args.horizons:
        if k not in tf_per_k:
            continue
        row = [f'{k:>3}']
        for n in cls_order:
            cid = OBJECT_TO_IDX.get(n)
            v = tf_per_k[k].get(cid, {}).get('acc')
            row.append(f'{v:>7.3f}' if v is not None else f'{"-":>7}')
        print('  ' + '  '.join(row))

    result = {
        'model_path': args.model_path,
        'trans_model_type': trans_kind,
        'env_name': margs.env_name,
        'horizons': args.horizons,
        'n_rollouts': args.n_rollouts,
        'teacher_forced_per_horizon': {
            str(k): {str(c): v for c, v in d.items()}
            for k, d in tf_per_k.items()
        },
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved {args.output_json}')


if __name__ == '__main__':
    main()
