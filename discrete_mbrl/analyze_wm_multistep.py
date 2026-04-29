"""
Multi-step World-Model Semantic Accuracy Analysis
==================================================
Extends analyze_wm_semantic_accuracy.py to compute k-step rollout accuracy
at multiple horizons (k in {1, 3, 5, 10}) for probe-vs-WM correlation.

Rollout protocol (free-running, action-conditioned):
    codes_0  = encode(obs_0)                       # ground truth initial codes
    codes_1_hat = WM.step(codes_0,    action_0)   # 1-step prediction
    codes_2_hat = WM.step(codes_1_hat, action_1)  # 2-step (feeds prediction)
    ...
Ground truth at each step comes from encode(env.step(action)). Accuracy is
measured per (token, class) where class is the SEMANTIC LABEL at the start
of the rollout — so a class's score reflects how well the WM preserves
information about that class across k steps.

Usage:
    python analyze_wm_multistep.py \\
        --model_path path/to/best_model.pt \\
        --n_rollouts 500 \\
        --horizons 1 3 5 10 \\
        --device cuda \\
        --output_json results.json

Supports both discrete and continuous transition models.
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

from env_helpers import make_env, preprocess_obs, OBJECT_TO_IDX

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
N_CLASSES = 11
OBJECT_NAMES = [IDX_TO_OBJECT.get(i, f'class_{i}') for i in range(N_CLASSES)]


# ── Model loading ──────────────────────────────────────────────────────────

def load_checkpoint(model_path, device):
    """Load ae_model + trans_model. Detects discrete vs continuous WM."""
    print(f"Loading: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    margs = types.SimpleNamespace(**ckpt['args'])

    from model_construction import make_ae
    from shared.models.encoder_models import VQVAEModel

    env = make_env(margs.env_name)
    reset_result = env.reset()
    sample_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([sample_obs])

    ae_model_type = getattr(margs, 'ae_model_type', 'vqvae')
    embedding_dim = getattr(margs, 'embedding_dim', 64)
    filter_size   = getattr(margs, 'filter_size', 8)
    ae_version    = str(getattr(margs, 'ae_model_version', '5'))

    encoder, decoder = make_ae(
        sample_obs.shape[1:], embedding_dim, filter_size, version=ae_version,
    )

    if ae_model_type == 'vqvae':
        ae_model = VQVAEModel(
            sample_obs.shape[1:],
            codebook_size=getattr(margs, 'codebook_size', 64),
            embedding_dim=embedding_dim,
            encoder=encoder, decoder=decoder,
            commitment_cost=0.25, ema_decay=0.99,
            dead_code_threshold=getattr(margs, 'dead_code_threshold', 0.0),
        )
        trans_kind = 'discrete'
    elif ae_model_type == 'vae_spatial':
        from shared.models.encoder_models import AEModelSpatial
        ae_model = AEModelSpatial(
            sample_obs.shape[1:],
            embedding_dim=embedding_dim,
            encoder=encoder, decoder=decoder, stochastic=True,
        )
        trans_kind = 'continuous'
    elif ae_model_type in ('vae', 'ae'):
        from shared.models.encoder_models import AEModel
        ae_model = AEModel(
            sample_obs.shape[1:],
            latent_dim=getattr(margs, 'latent_dim', None),
            encoder=encoder, decoder=decoder,
            stochastic=(ae_model_type == 'vae'),
        )
        trans_kind = 'continuous'
    else:
        raise ValueError(f"Unsupported ae_model_type: {ae_model_type}")

    ae_model.load_state_dict(ckpt['ae_model_state_dict'])
    ae_model = ae_model.to(device).eval()

    # ── Transition model ──
    trans_model = None
    trans_model_type = getattr(margs, 'trans_model_type', trans_kind)
    if 'trans_model_state_dict' in ckpt and getattr(margs, 'use_world_model', False):
        try:
            if trans_model_type == 'discrete':
                from shared.models.transition_models import DiscreteTransitionModel
                n_latents = ae_model.n_latent_embeds
                n_codes   = getattr(margs, 'codebook_size', 64)
                hidden    = getattr(margs, 'trans_hidden', 256)
                depth     = getattr(margs, 'trans_depth', 3)
                trans_model = DiscreteTransitionModel(
                    n_latents, n_codes, embedding_dim,
                    env.action_space,
                    hidden_sizes=[hidden] * depth,
                    stochastic=getattr(margs, 'stochastic', False),
                    stoch_hidden_sizes=[256, 256],
                    discretizer_hidden_sizes=[256],
                    use_soft_embeds=False,
                    return_logits=False,
                )
            else:  # continuous
                from shared.models.transition_models import ContinuousTransitionModel
                latent_dim = ae_model.latent_dim
                hidden = getattr(margs, 'trans_hidden', 256)
                depth  = getattr(margs, 'trans_depth', 3)
                trans_model = ContinuousTransitionModel(
                    latent_dim, env.action_space,
                    hidden_sizes=[hidden] * depth,
                    stochastic=getattr(margs, 'stochastic', None),
                    stoch_hidden_sizes=[256, 256],
                    discretizer_hidden_sizes=[256],
                )
            trans_model.load_state_dict(ckpt['trans_model_state_dict'])
            trans_model = trans_model.to(device).eval()
            print(f"Transition model ({trans_model_type}) loaded "
                  f"({sum(p.numel() for p in trans_model.parameters()):,} params)")
        except Exception as e:
            print(f"  ⚠ Could not load transition model: {e}")
            trans_model = None

    print(f"Encoder {ae_model_type} v{ae_version}, "
          f"ae_type={ae_model_type}, trans_type={trans_model_type}")
    return ae_model, trans_model, trans_model_type, env, margs


# ── Helpers ────────────────────────────────────────────────────────────────

def semantic_grid(env, n_lat_side):
    """Return (L,) int tensor of semantic class ids at current env state."""
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
    return torch.from_numpy(ge_rowmajor.flatten().copy()).long()


def encode_codes(ae_model, obs_t, device):
    """Encode single obs → (L,) discrete code indices on CPU, or None if continuous."""
    obs_in = obs_t.unsqueeze(0).to(device)
    with torch.no_grad():
        try:
            codes = ae_model.encode(obs_in, return_quantized=False)
            if codes.dim() == 3:
                codes = codes.view(1, -1)
            return codes.long().squeeze(0).cpu()  # (L,)
        except TypeError:
            return None


def encode_latent(ae_model, obs_t, device):
    """Encode single obs → (latent_dim,) continuous latent."""
    obs_in = obs_t.unsqueeze(0).to(device)
    with torch.no_grad():
        try:
            latent = ae_model.encode(obs_in, return_quantized=True)
        except TypeError:
            latent = ae_model.encode(obs_in)
        return latent.view(-1)  # (D*L,) for vae, or (L*emb,) for vqvae


# ── Multi-step rollout ─────────────────────────────────────────────────────

def rollout_discrete(trans_model, codes_0, actions_seq, device):
    """Free-running discrete rollout.
    codes_0:      (L,) int
    actions_seq:  (K,) int
    Returns:      list of length K; each (L,) int predicted codes at t+1..t+K
    """
    trans_model.eval()
    preds = []
    codes = codes_0.unsqueeze(0).to(device)  # (1, L)
    for k in range(len(actions_seq)):
        act = torch.tensor([actions_seq[k]], device=device, dtype=torch.long)
        with torch.no_grad():
            out = trans_model(codes, act, return_logits=True)
            # DiscreteTransitionModel returns (state_logits, reward, gamma)
            state_logits = out[0]
            # state_logits: (B, n_codes, L)
            pred = state_logits.argmax(dim=1)  # (1, L)
        preds.append(pred.squeeze(0).cpu())
        codes = pred
    return preds


def rollout_continuous(trans_model, ae_model, latent_0, actions_seq, device):
    """Free-running continuous rollout.
    latent_0:     (D,) float
    Returns:      list of length K; each (L,) int codes decoded via nearest-neighbor
                  OR (L,) float VAE mean vectors — we return nothing by default,
                  matching accuracy is computed against ground-truth latents externally.
    Actually, for semantic accuracy we want to turn continuous latents into
    per-token predictions. We do this by taking top-1 via the probe decision
    boundary. Since the VAE has no codes, we return the latent sequence.
    """
    trans_model.eval()
    preds = []
    latent = latent_0.unsqueeze(0).to(device)  # (1, D)
    for k in range(len(actions_seq)):
        act = torch.tensor([actions_seq[k]], device=device, dtype=torch.long)
        with torch.no_grad():
            states, _, _ = trans_model(latent, act)
        preds.append(states.squeeze(0).cpu())
        latent = states
    return preds  # list of (D,) tensors


# ── Main: multi-step per-class accuracy ────────────────────────────────────

def compute_multistep_accuracy(ae_model, trans_model, trans_kind, env,
                                n_rollouts, horizons, device, n_lat_side,
                                max_steps=None):
    """
    For each rollout (fresh env reset, random actions):
      - encode ground-truth codes at t=0..max(horizons)
      - unroll WM predictions from codes_0 using same actions
      - record per-class accuracy at each horizon k
    Accuracy: fraction of TOKEN positions where prediction matches ground-truth
    code, grouped by semantic class label at time 0.
    """
    if max_steps is None:
        max_steps = max(horizons)

    # Accumulators: per horizon, per class, correct_count + total_count
    stats = {k: {c: [0, 0] for c in range(N_CLASSES)} for k in horizons}

    for roll_idx in range(n_rollouts):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs_t = torch.from_numpy(obs).float()

        # Encode initial state
        if trans_kind == 'discrete':
            codes_seq = [encode_codes(ae_model, obs_t, device)]  # list of (L,) int
            latent_seq = None
        else:
            latent_seq = [encode_latent(ae_model, obs_t, device).cpu()]
            codes_seq = None

        sem_labels_0 = semantic_grid(env, n_lat_side)  # (L,) int

        actions = []
        valid = True
        for step in range(max_steps):
            a = env.action_space.sample()
            actions.append(a)
            step_result = env.step(a)
            obs_next = step_result[0] if isinstance(step_result, tuple) else step_result
            done = step_result[2] if len(step_result) >= 3 else False

            obs_next_t = torch.from_numpy(obs_next).float()
            if trans_kind == 'discrete':
                codes_seq.append(encode_codes(ae_model, obs_next_t, device))
            else:
                latent_seq.append(encode_latent(ae_model, obs_next_t, device).cpu())

            if done:
                # Truncate this rollout if episode ends early; only use horizons
                # reachable. We still count up to (step+1).
                break

        reached = step + 1
        if reached < 1:
            continue

        # Run WM rollout with same actions
        if trans_kind == 'discrete':
            preds = rollout_discrete(trans_model, codes_seq[0], actions[:reached], device)
            gt_codes = codes_seq[1:reached + 1]  # list of (L,) int

            for k in horizons:
                if k > reached:
                    continue
                pred_k = preds[k - 1]
                gt_k = gt_codes[k - 1]
                correct_mask = (pred_k == gt_k)  # (L,) bool
                for c in range(N_CLASSES):
                    class_mask = (sem_labels_0 == c)
                    n = class_mask.sum().item()
                    if n == 0:
                        continue
                    nc = (correct_mask & class_mask).sum().item()
                    stats[k][c][0] += nc
                    stats[k][c][1] += n
        else:
            # Continuous: we compare predicted latent vs ground-truth latent via
            # MSE-per-token-slot against a threshold of "closer to GT than to the
            # mean of other latents in the batch". For a per-class accuracy
            # analogue, we cosine-similarity the per-token chunks of the latent.
            # Token chunking assumes latent_dim = L * emb_dim.
            latent_dim = latent_seq[0].shape[0]
            L_tokens = n_lat_side * n_lat_side
            emb_dim = latent_dim // L_tokens
            preds = rollout_continuous(trans_model, ae_model,
                                        latent_seq[0].to(device),
                                        actions[:reached], device)

            for k in horizons:
                if k > reached:
                    continue
                pred_k = preds[k - 1].view(L_tokens, emb_dim)       # (L, D)
                gt_k   = latent_seq[k].view(L_tokens, emb_dim)       # (L, D)
                # Normalized: a token is "correct" if its cosine sim to the
                # true token is higher than to any other token in the same frame
                # (i.e. argmax of similarity matrix is the diagonal).
                pred_n = F.normalize(pred_k, dim=-1)
                gt_n   = F.normalize(gt_k, dim=-1)
                sim = pred_n @ gt_n.T  # (L, L)
                pred_argmax = sim.argmax(dim=1)   # (L,)
                correct_mask = (pred_argmax == torch.arange(L_tokens))

                for c in range(N_CLASSES):
                    class_mask = (sem_labels_0 == c)
                    n = class_mask.sum().item()
                    if n == 0:
                        continue
                    nc = (correct_mask & class_mask).sum().item()
                    stats[k][c][0] += nc
                    stats[k][c][1] += n

        if (roll_idx + 1) % 50 == 0:
            print(f"  rollouts: {roll_idx + 1}/{n_rollouts}")

    # Convert to accuracy dicts
    result = {}
    for k in horizons:
        per_class = {}
        for c in range(N_CLASSES):
            n_total = stats[k][c][1]
            if n_total < 10:
                continue
            acc = stats[k][c][0] / n_total
            per_class[c] = {
                'acc': acc,
                'count': n_total,
                'name': OBJECT_NAMES[c],
            }
        result[k] = per_class
    return result


# ── Semantic probe (reuse from analyze_wm_semantic_accuracy) ───────────────

def compute_probe_accuracy_per_class(ae_model, env, n_frames, device, n_lat_side):
    from sklearn.linear_model import LogisticRegression

    ae_model.eval()
    emb_list = []
    sem_list = []

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs_t = torch.from_numpy(obs).float()

    collected = 0
    while collected < n_frames:
        with torch.no_grad():
            obs_in = obs_t.unsqueeze(0).to(device)
            try:
                flat_emb = ae_model.encode(obs_in, return_quantized=True)
            except TypeError:
                flat_emb = ae_model.encode(obs_in)

        latent_dim = flat_emb.shape[-1]
        L_tokens = n_lat_side * n_lat_side
        emb_dim = latent_dim // L_tokens
        per_token = flat_emb.view(L_tokens, emb_dim).cpu()

        sem_t = semantic_grid(env, n_lat_side)

        emb_list.append(per_token.numpy())
        sem_list.append(sem_t.numpy())

        collected += 1
        action = env.action_space.sample()
        step_result = env.step(action)
        obs = step_result[0] if isinstance(step_result, tuple) else step_result
        done = step_result[2] if len(step_result) >= 3 else False
        obs_t = torch.from_numpy(obs).float()
        if done:
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs_t = torch.from_numpy(obs).float()

    X = np.concatenate(emb_list, axis=0)
    y = np.concatenate(sem_list, axis=0)

    probe = LogisticRegression(max_iter=300, class_weight='balanced', C=1.0)
    probe.fit(X, y)
    y_pred = probe.predict(X)

    per_class = {}
    for c in np.unique(y):
        mask = y == c
        count = mask.sum()
        if count < 10:
            continue
        acc = (y_pred[mask] == c).mean()
        per_class[int(c)] = {'acc': float(acc), 'count': int(count),
                             'name': OBJECT_NAMES[int(c)]}
    return per_class


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_rollouts', type=int, default=500)
    parser.add_argument('--probe_frames', type=int, default=20000)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--skip_probe', action='store_true',
                        help="Skip the semantic probe (just run WM)")
    args = parser.parse_args()

    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(
        args.model_path, args.device)
    n_lat = getattr(ae_model, 'n_latent_embeds', None)
    if n_lat is None:
        n_lat = ae_model.latent_dim // getattr(margs, 'embedding_dim', 64)
    n_lat_side = int(round(n_lat ** 0.5))

    # ── Probe ──
    probe_acc = {}
    if not args.skip_probe:
        print(f"\nRunning semantic probe ({args.probe_frames} frames)...")
        env2 = make_env(margs.env_name)
        probe_acc = compute_probe_accuracy_per_class(
            ae_model, env2, args.probe_frames, args.device, n_lat_side)

    # ── Multi-step WM ──
    wm_multi = {}
    if trans_model is not None:
        print(f"\nMulti-step WM rollouts: {args.n_rollouts} × up to {max(args.horizons)} steps")
        wm_multi = compute_multistep_accuracy(
            ae_model, trans_model, trans_kind, env,
            args.n_rollouts, args.horizons, args.device, n_lat_side)
    else:
        print("\n⚠ No transition model in checkpoint — skipping WM.")

    # ── Report ──
    print("\n" + "=" * 80)
    print("MULTI-STEP PER-CLASS ACCURACY")
    print("=" * 80)
    header = f"{'Class':<12} {'Probe':>8}"
    for k in args.horizons:
        header += f"  {'WM-'+str(k)+'step':>9}"
    header += f"  {'Count':>8}"
    print(header)
    print("-" * len(header))

    all_classes = sorted(set(list(probe_acc.keys()) +
                              [c for k in wm_multi.values() for c in k.keys()]))
    rows = []
    for c in all_classes:
        p = probe_acc.get(c, {})
        p_acc = p.get('acc', None)
        name = OBJECT_NAMES[c]
        row = {'class': c, 'name': name,
                'probe_acc': None if p_acc is None else round(p_acc, 4),
                'count': p.get('count', 0)}
        line = f"  {name:<10} "
        line += f"{p_acc*100:>6.1f}%" if p_acc is not None else f"{'—':>7}"

        for k in args.horizons:
            w = wm_multi.get(k, {}).get(c, {})
            w_acc = w.get('acc', None)
            row[f'wm_{k}step'] = None if w_acc is None else round(w_acc, 4)
            line += f"  {w_acc*100:>8.1f}%" if w_acc is not None else f"  {'—':>9}"
            if row['count'] == 0 and w.get('count'):
                row['count'] = w['count']
        line += f"  {row['count']:>8,}"
        print(line)
        rows.append(row)

    # Correlations: probe vs each horizon
    corrs = {}
    for k in args.horizons:
        xs, ys = [], []
        for r in rows:
            if r['probe_acc'] is not None and r[f'wm_{k}step'] is not None:
                xs.append(r['probe_acc']); ys.append(r[f'wm_{k}step'])
        if len(xs) >= 3:
            corrs[k] = float(np.corrcoef(xs, ys)[0, 1])
        else:
            corrs[k] = None

    print("\nPearson r (probe vs WM-k-step):")
    for k in args.horizons:
        r = corrs[k]
        s = f"{r:+.3f}" if r is not None else "—"
        print(f"  k={k:2d}: {s}")

    # ── Save ──
    result = {
        'model_path': args.model_path,
        'encoder_version': str(getattr(margs, 'ae_model_version', '?')),
        'ae_model_type':   getattr(margs, 'ae_model_type', '?'),
        'trans_model_type': trans_kind,
        'env_name':     margs.env_name,
        'horizons':     args.horizons,
        'probe_acc_per_class': {
            str(c): v for c, v in probe_acc.items()
        },
        'wm_acc_per_class_per_horizon': {
            str(k): {str(c): v for c, v in wm_multi[k].items()}
            for k in wm_multi
        },
        'rows':         rows,
        'pearson_r_per_horizon': {str(k): v for k, v in corrs.items()},
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output_json}")

    return result


if __name__ == '__main__':
    main()
