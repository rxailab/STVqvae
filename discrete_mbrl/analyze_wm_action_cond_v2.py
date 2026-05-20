"""
Phase 1 analyzer for "Closing the Probe–WM Gap" (follow-up paper).

Extends analyze_wm_action_cond.py with four per-class action-dependence metrics:

  per_class_action_diff_norm  {c: float}
    For each MiniGrid object class c, mean ||T(z,a)[p] - T(z,a')[p]|| over
    positions p where the *current* semantic label equals c, and over a≠a'.
    Tells us whether the WM's action-collapse is class-specific (we expect
    near-zero on door/key positions, non-zero on wall/empty).

  per_class_state_change_norm {c: float}
    Per-class baseline: mean ||T(z,a)[p] - z[p]|| restricted to positions
    of class c. Required to normalise per_class_action_diff_norm.

  per_class_action_dep_ratio  {c: float}
    per_class_action_diff_norm[c] / max(per_class_state_change_norm[c], 1e-9).
    Headline metric for the next paper: action-dependence broken down by
    semantic class.

  epsilon_action_collapse     float
    Fraction of (state, a, a') pairs where ||T(z,a) - T(z,a')|| <
    0.1 * state_change_norm[state]. Soft proxy for "WM ignores action choice".

  action_residual_rank        float
    Effective rank of {T(z, a) - mean_a T(z, a)} (mean-action-subtracted),
    instead of the existing T(z, a) - z form. Cleaner measure of
    action-conditioned signal because it removes the action-independent
    state drift.

Backward-compatible: keeps every key from the original analyzer in the
output JSON.

Usage:
    python analyze_wm_action_cond_v2.py --model_path ckpt.pt --output_json out.json
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from analyze_wm_multistep import load_checkpoint, encode_codes
from analyze_wm_phaseB_e8 import build_mlp, encode_flat
from env_helpers import OBJECT_TO_IDX

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
N_CLASSES = 11
OBJECT_NAMES = [IDX_TO_OBJECT.get(i, f'class_{i}') for i in range(N_CLASSES)]


def _semantic_grid(env, n_lat_side):
    """Return semantic-class grid flattened to length L = n_lat_side**2.

    Mirrors analyze_wm_semantic_accuracy.collect_transitions, which is the
    canonical extractor used in §4 of the dissociation paper.
    """
    ug = env.unwrapped
    ge = ug.grid.encode()[:, :, 0].copy()
    ge[ug.agent_pos[0], ug.agent_pos[1]] = OBJECT_TO_IDX['agent']
    ge_rowmajor = ge.T
    if ge_rowmajor.shape != (n_lat_side, n_lat_side):
        from PIL import Image as PIL_Image
        ge_rowmajor = np.array(
            PIL_Image.fromarray(ge_rowmajor.astype(np.uint8)).resize(
                (n_lat_side, n_lat_side), PIL_Image.NEAREST
            )
        )
    return ge_rowmajor.flatten().astype(np.int64)


def collect_states(env, ae_model, policy, n_steps, device, trans_kind, n_lat_side):
    """Like the original collector, plus per-state per-position semantic labels."""
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    obs_t = torch.from_numpy(obs).float()
    z_list, code_list, sem_list = [], [], []
    n_actions = env.action_space.n
    for t in range(n_steps):
        sem = _semantic_grid(env, n_lat_side)
        sem_list.append(sem)

        z = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
        z_list.append(z.astype(np.float32))
        if trans_kind == 'discrete':
            c = encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
            code_list.append(c.astype(np.int64))
        if policy is not None:
            with torch.no_grad():
                logits = policy(torch.from_numpy(z).unsqueeze(0).to(device))
                probs = F.softmax(logits.squeeze(0), dim=-1)
                a = int(torch.multinomial(probs, 1).item())
        else:
            a = int(np.random.randint(n_actions))
        sr = env.step(a)
        obs_next = sr[0] if isinstance(sr, tuple) else sr
        done = sr[2] if len(sr) >= 3 else False
        obs_t = torch.from_numpy(obs_next).float()
        if done:
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
            obs_t = torch.from_numpy(obs).float()
    return (np.stack(z_list),
            (np.stack(code_list) if code_list else None),
            np.stack(sem_list))


def _compute_Tza(ae_model, trans_model, trans_kind, z_arr, code_arr, n_actions, device, bs=256):
    """Compute T(z, a) and per-action reward/gamma for every (state, action)."""
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    input_dim = L * embedding_dim
    Tza   = np.zeros((len(z_arr), n_actions, input_dim), dtype=np.float32)
    R     = np.zeros((len(z_arr), n_actions), dtype=np.float32)
    Gamma = np.zeros((len(z_arr), n_actions), dtype=np.float32)

    if trans_kind == 'discrete':
        cb = ae_model.quantizer._embedding.weight.detach()
        codes_t = torch.from_numpy(code_arr).long().to(device)
        for i in range(0, len(z_arr), bs):
            c_b = codes_t[i:i + bs]
            B = c_b.shape[0]
            c_rep = c_b.unsqueeze(1).repeat(1, n_actions, 1).view(B * n_actions, -1)
            a_rep = torch.arange(n_actions, device=device).unsqueeze(0).repeat(B, 1).view(-1)
            with torch.no_grad():
                logits, r, g = trans_model(c_rep, a_rep, return_logits=True)
                next_codes = logits.argmax(dim=1)
                z_next = cb[next_codes].permute(0, 2, 1).reshape(B * n_actions, L * embedding_dim)
            Tza[i:i + bs]   = z_next.view(B, n_actions, -1).cpu().numpy()
            R[i:i + bs]     = (r if r.dim() <= 2 else r.view(B * n_actions, -1)[:, 0]).view(B, n_actions).cpu().numpy()
            Gamma[i:i + bs] = (g if g.dim() <= 2 else g.view(B * n_actions, -1)[:, 0]).view(B, n_actions).cpu().numpy()
    else:
        z_t = torch.from_numpy(z_arr).to(device)
        for i in range(0, len(z_arr), bs):
            z_b = z_t[i:i + bs]
            B = z_b.shape[0]
            z_rep = z_b.unsqueeze(1).repeat(1, n_actions, 1).view(B * n_actions, -1)
            a_rep = torch.arange(n_actions, device=device).unsqueeze(0).repeat(B, 1).view(-1)
            with torch.no_grad():
                zn, r, g = trans_model(z_rep, a_rep)
                zn_flat = zn.view(B * n_actions, -1)
            Tza[i:i + bs]   = zn_flat.view(B, n_actions, -1).cpu().numpy()
            R[i:i + bs]     = r.view(B, n_actions).cpu().numpy()
            Gamma[i:i + bs] = g.view(B, n_actions).cpu().numpy()
    return Tza, R, Gamma, L, embedding_dim


def _per_class_metrics(Tza, z_arr, sem_arr, L, embedding_dim):
    """Per-class action-diff and state-change norms.

    Tza:  (N, A, L*D)
    z_arr:(N, L*D)
    sem_arr: (N, L) int semantic class per position
    """
    N, A, _ = Tza.shape
    D = embedding_dim
    Tza_4d = Tza.reshape(N, A, L, D)        # (N, A, L, D)
    z_3d   = z_arr.reshape(N, L, D)         # (N, L, D)

    # Per-position action-diff norm averaged over a<a' pairs.
    # Vectorise over action pairs by computing the pairwise mean directly:
    #   mean_{a<a'} ||Tza[a,p] - Tza[a',p]||
    # Compute per-position per-pair distances.
    pair_means = np.zeros((N, L), dtype=np.float64)
    pair_count = 0
    for a in range(A):
        for ap in range(a + 1, A):
            d = np.linalg.norm(Tza_4d[:, a] - Tza_4d[:, ap], axis=2)  # (N, L)
            pair_means += d
            pair_count += 1
    pair_means /= max(pair_count, 1)        # (N, L) mean over action pairs

    # Per-position per-action state-change norm averaged over a.
    state_chg_pp = np.linalg.norm(Tza_4d - z_3d[:, None, :, :], axis=3).mean(axis=1)  # (N, L)

    per_class_diff = {}
    per_class_chg  = {}
    per_class_ratio = {}
    per_class_count = {}
    for c in range(N_CLASSES):
        mask = (sem_arr == c)                # (N, L) bool
        n_pos = int(mask.sum())
        if n_pos == 0:
            continue
        per_class_count[c] = n_pos
        per_class_diff[c] = float(pair_means[mask].mean())
        per_class_chg[c]  = float(state_chg_pp[mask].mean())
        per_class_ratio[c] = per_class_diff[c] / max(per_class_chg[c], 1e-9)

    return per_class_diff, per_class_chg, per_class_ratio, per_class_count


def _epsilon_action_collapse(Tza, z_arr, eps_frac=0.1):
    """Fraction of (state, a, a') pairs where the two predictions are within
    eps_frac * mean-state-change of each other."""
    N, A, D = Tza.shape
    state_chg = np.linalg.norm(Tza - z_arr[:, None, :], axis=2)     # (N, A)
    state_chg_mean = state_chg.mean(axis=1, keepdims=True)          # (N, 1)
    threshold = eps_frac * state_chg_mean.squeeze(1)                # (N,)
    collapsed = 0
    total = 0
    for a in range(A):
        for ap in range(a + 1, A):
            d = np.linalg.norm(Tza[:, a] - Tza[:, ap], axis=1)      # (N,)
            collapsed += int((d < threshold).sum())
            total += N
    return float(collapsed) / max(total, 1)


def _action_residual_rank(Tza, n_subsample=500):
    """SVD-rank of action-conditioned residual after subtracting the
    mean-action prediction. Cleaner than (T(z,a) - z) because it removes
    the action-independent drift component."""
    N, A, D = Tza.shape
    sub = np.random.choice(N, size=min(n_subsample, N), replace=False)
    eff_ranks = []
    for i in sub:
        m = Tza[i] - Tza[i].mean(axis=0, keepdims=True)  # (A, D)
        if np.linalg.norm(m) < 1e-9:
            eff_ranks.append(0)
            continue
        tol = 0.01 * np.linalg.norm(m, axis=1).mean()
        s = np.linalg.svd(m, compute_uv=False)
        eff_ranks.append(int((s > tol).sum()))
    return float(np.mean(eff_ranks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_states', type=int, default=2000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Loading {args.model_path}')
    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_model is None:
        raise RuntimeError('No transition model in checkpoint.')

    L = ae_model.n_latent_embeds
    n_lat_side = int(round(np.sqrt(L)))
    if n_lat_side * n_lat_side != L:
        raise RuntimeError(f'n_latent_embeds={L} is not a perfect square; per-class breakdown needs a square grid.')
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    input_dim = L * embedding_dim
    n_actions = env.action_space.n

    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    policy = None
    if 'policy_state_dict' in src and isinstance(src.get('args'), dict) and 'policy_hidden' in src['args']:
        try:
            policy = build_mlp(input_dim, src['args'].get('policy_hidden', [256, 256]),
                               n_actions, 'relu').to(args.device).eval()
            policy.load_state_dict(src['policy_state_dict'])
        except Exception as e:
            print(f'  (policy load failed: {e}; falling back to random-action state collection)')
            policy = None

    print(f'\n[1/2] Collecting {args.n_states} states '
          f'({"policy" if policy is not None else "random"} actions; '
          f'trans_kind={trans_kind}; n_lat_side={n_lat_side})')
    z_arr, code_arr, sem_arr = collect_states(env, ae_model, policy, args.n_states,
                                              args.device, trans_kind, n_lat_side)
    print(f'  z shape: {z_arr.shape}  sem shape: {sem_arr.shape}  n_actions: {n_actions}')

    print(f'\n[2/2] Computing T(z, a) for {len(z_arr)} × {n_actions} action pairs')
    Tza, R, Gamma, L_chk, D_chk = _compute_Tza(ae_model, trans_model, trans_kind,
                                               z_arr, code_arr, n_actions, args.device)
    assert L_chk == L and D_chk == embedding_dim

    # Original metrics (preserve backward compat) ───────────────────────────
    z_broadcast = z_arr[:, None, :]
    state_change = np.linalg.norm(Tza - z_broadcast, axis=2)            # (N, A)
    state_change_norm = float(state_change.mean())

    pair_dists = []
    for a in range(n_actions):
        for ap in range(a + 1, n_actions):
            pair_dists.append(np.linalg.norm(Tza[:, a] - Tza[:, ap], axis=1))
    pair_dists = np.concatenate(pair_dists)
    action_diff_norm = float(pair_dists.mean())
    action_dep_ratio = action_diff_norm / max(state_change_norm, 1e-9)

    reward_action_var = float(R.var(axis=1).mean())
    gamma_action_var  = float(Gamma.var(axis=1).mean())
    reward_action_range = float((R.max(axis=1) - R.min(axis=1)).mean())
    gamma_action_range  = float((Gamma.max(axis=1) - Gamma.min(axis=1)).mean())

    # Original effective_action_rank (T(z,a) - z, for backward compat)
    deltas = Tza - z_broadcast
    sub = np.random.choice(len(deltas), size=min(500, len(deltas)), replace=False)
    eff_ranks = []
    for i in sub:
        m = deltas[i]
        tol = 0.01 * state_change[i].mean()
        s = np.linalg.svd(m, compute_uv=False)
        eff_ranks.append(int((s > tol).sum()))
    effective_action_rank = float(np.mean(eff_ranks))

    # New metrics ───────────────────────────────────────────────────────────
    per_class_diff, per_class_chg, per_class_ratio, per_class_count = (
        _per_class_metrics(Tza, z_arr, sem_arr, L, embedding_dim))
    epsilon_action_collapse = _epsilon_action_collapse(Tza, z_arr, eps_frac=0.1)
    action_residual_rank = _action_residual_rank(Tza)

    print(f'\n=== Action-conditioning probe v2 ===')
    print(f'  state_change_norm    = {state_change_norm:.4f}')
    print(f'  action_diff_norm     = {action_diff_norm:.4f}')
    print(f'  action_dep_ratio     = {action_dep_ratio:.4f}')
    print(f'  reward_action_var    = {reward_action_var:.6f}')
    print(f'  gamma_action_var     = {gamma_action_var:.6f}')
    print(f'  effective_action_rank= {effective_action_rank:.2f} / {n_actions}')
    print(f'  action_residual_rank = {action_residual_rank:.2f} / {n_actions}  (mean-subtracted)')
    print(f'  epsilon_action_collapse(eps=0.1) = {epsilon_action_collapse:.3f}')
    print(f'  --- per-class action_dep_ratio ---')
    for c, r in sorted(per_class_ratio.items()):
        print(f'    [{c:2d} {OBJECT_NAMES[c]:>8s}] ratio={r:.3f}  '
              f'diff={per_class_diff[c]:.3f}  chg={per_class_chg[c]:.3f}  n={per_class_count[c]}')

    result = {
        'model_path': args.model_path,
        'env_name': margs.env_name,
        'n_actions': int(n_actions),
        'n_states': int(len(z_arr)),
        'state_change_norm': state_change_norm,
        'action_diff_norm': action_diff_norm,
        'action_dep_ratio': action_dep_ratio,
        'reward_action_var': reward_action_var,
        'gamma_action_var': gamma_action_var,
        'reward_action_range': reward_action_range,
        'gamma_action_range': gamma_action_range,
        'effective_action_rank': effective_action_rank,
        'action_residual_rank': action_residual_rank,
        'epsilon_action_collapse': epsilon_action_collapse,
        'per_class_action_diff_norm':  {str(c): v for c, v in per_class_diff.items()},
        'per_class_state_change_norm': {str(c): v for c, v in per_class_chg.items()},
        'per_class_action_dep_ratio':  {str(c): v for c, v in per_class_ratio.items()},
        'per_class_count':             {str(c): v for c, v in per_class_count.items()},
        'class_names': OBJECT_NAMES,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved {args.output_json}')


if __name__ == '__main__':
    main()
