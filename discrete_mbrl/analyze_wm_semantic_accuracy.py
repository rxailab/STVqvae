"""
Phase 5: World-Model Semantic Accuracy Analysis
================================================
For each encoder checkpoint, measures the correlation between:
  - Semantic probe accuracy per class  (how well a linear probe classifies tokens)
  - World-model transition accuracy per class  (how well the WM predicts next tokens
    at positions occupied by each semantic class)

Usage:
    python analyze_wm_semantic_accuracy.py \\
        --model_path path/to/best_model.pt \\
        --n_frames 5000 \\
        --device cuda \\
        --output_json results.json

Output (stdout + JSON):
    Per-class:  probe_acc, wm_acc, n_samples
    Overall:    correlation coefficient
    Scatter plot data for the paper figure
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
    """Load ae_model + trans_model from a PPO checkpoint."""
    print(f"Loading: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    margs = types.SimpleNamespace(**ckpt['args'])

    from model_construction import make_ae
    from shared.models.encoder_models import VQVAEModel

    env = make_env(margs.env_name)
    reset_result = env.reset()
    sample_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([sample_obs])

    encoder, decoder = make_ae(
        sample_obs.shape[1:],
        getattr(margs, 'embedding_dim', 64),
        getattr(margs, 'filter_size', 8),
        version=str(getattr(margs, 'ae_model_version', '5')),
    )
    ae_model = VQVAEModel(
        sample_obs.shape[1:],
        codebook_size=getattr(margs, 'codebook_size', 64),
        embedding_dim=getattr(margs, 'embedding_dim', 64),
        encoder=encoder, decoder=decoder,
        commitment_cost=0.25, ema_decay=0.99,
        dead_code_threshold=getattr(margs, 'dead_code_threshold', 0.0),
    )
    ae_model.load_state_dict(ckpt['ae_model_state_dict'])
    ae_model = ae_model.to(device).eval()

    # ── Transition model (build directly from saved args) ──
    trans_model = None
    if 'trans_model_state_dict' in ckpt and getattr(margs, 'use_world_model', False):
        try:
            from shared.models.transition_models import DiscreteTransitionModel
            n_latents  = ae_model.n_latent_embeds
            n_codes    = getattr(margs, 'codebook_size', 64)
            emb_dim    = getattr(margs, 'embedding_dim', 64)
            hidden     = getattr(margs, 'trans_hidden', 256)
            depth      = getattr(margs, 'trans_depth', 3)
            stochastic = getattr(margs, 'stochastic', False)
            trans_model = DiscreteTransitionModel(
                n_latents, n_codes, emb_dim,
                env.action_space,   # pass full space so act_dim = space.n
                hidden_sizes=[hidden] * depth,
                stochastic=stochastic,
                stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256],
                use_soft_embeds=False,
                return_logits=False,
            )
            trans_model.load_state_dict(ckpt['trans_model_state_dict'])
            trans_model = trans_model.to(device).eval()
            print(f"Transition model loaded ({sum(p.numel() for p in trans_model.parameters()):,} params)")
        except Exception as e:
            print(f"  ⚠ Could not load transition model: {e}")
            trans_model = None

    print(f"Encoder v{getattr(margs, 'ae_model_version', '?')}, "
          f"codebook={getattr(margs, 'codebook_size', 64)}, "
          f"n_latents={ae_model.n_latent_embeds}")
    return ae_model, trans_model, env, margs


# ── Data collection ────────────────────────────────────────────────────────

def collect_transitions(env, ae_model, trans_model, n_frames, device, n_lat_side):
    """Collect (codes_t, action, codes_t1, sem_labels_t) tuples."""
    ae_model.eval()
    if trans_model is not None:
        trans_model.eval()

    codes_list     = []   # (N, n_lat) int
    actions_list   = []   # (N,) int
    codes_next_list = []  # (N, n_lat) int
    sem_list       = []   # (N, n_lat) int  semantic labels at t
    flat_emb_list  = []   # (N, n_lat*emb_dim) float — for WM prediction

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs_t = torch.from_numpy(obs).float()

    collected = 0
    while collected < n_frames:
        # Encode current frame
        with torch.no_grad():
            obs_in = obs_t.unsqueeze(0).to(device)
            flat_emb_t = ae_model.encode(obs_in, return_quantized=True)  # (1, L*D)
            codes_t = ae_model.encode(obs_in, return_quantized=False)    # (1, L) indices

        if codes_t.dim() == 3:   # (1, H, W) → (1, H*W)
            codes_t = codes_t.view(1, -1)
        codes_t = codes_t.long()

        # Semantic label grid
        ug = env.unwrapped
        ge = ug.grid.encode()[:, :, 0].copy()
        ge[ug.agent_pos[0], ug.agent_pos[1]] = OBJECT_TO_IDX['agent']
        ge_rowmajor = ge.T  # (H, W)
        if ge_rowmajor.shape != (n_lat_side, n_lat_side):
            from PIL import Image as PIL_Image
            ge_rowmajor = np.array(
                PIL_Image.fromarray(ge_rowmajor.astype(np.uint8)).resize(
                    (n_lat_side, n_lat_side), PIL_Image.NEAREST
                )
            )
        sem_t = torch.from_numpy(ge_rowmajor.flatten().copy()).long()  # (L,)

        # Step environment
        action = env.action_space.sample()
        step_result = env.step(action)
        obs_next = step_result[0] if isinstance(step_result, tuple) else step_result
        done = step_result[2] if len(step_result) >= 3 else False

        obs_next_t = torch.from_numpy(obs_next).float()

        # Encode next frame
        with torch.no_grad():
            obs_next_in = obs_next_t.unsqueeze(0).to(device)
            codes_t1 = ae_model.encode(obs_next_in, return_quantized=False)
        if codes_t1.dim() == 3:
            codes_t1 = codes_t1.view(1, -1)
        codes_t1 = codes_t1.long()

        codes_list.append(codes_t.cpu().squeeze(0))
        actions_list.append(action)
        codes_next_list.append(codes_t1.cpu().squeeze(0))
        sem_list.append(sem_t)
        flat_emb_list.append(flat_emb_t.cpu().squeeze(0))

        collected += 1
        if collected % 1000 == 0:
            print(f"  {collected}/{n_frames} frames")

        obs_t = obs_next_t
        if done:
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs_t = torch.from_numpy(obs).float()

    codes_all     = torch.stack(codes_list)       # (N, L)
    actions_all   = torch.tensor(actions_list)    # (N,)
    codes_next_all = torch.stack(codes_next_list) # (N, L)
    sem_all       = torch.stack(sem_list)          # (N, L)
    flat_emb_all  = torch.stack(flat_emb_list)    # (N, L*D)

    return codes_all, actions_all, codes_next_all, sem_all, flat_emb_all


# ── WM prediction ──────────────────────────────────────────────────────────

def compute_wm_accuracy_per_class(trans_model, codes_all, actions_all,
                                   codes_next_all, sem_all, device,
                                   batch_size=256):
    """Compute per-class WM 1-step prediction accuracy.

    For each token position, asks: does the WM predict the correct next code?
    Groups results by semantic class of the *current* token position.
    """
    trans_model.eval()
    N, L = codes_all.shape

    all_correct = []   # bool per (sample, latent) pair
    all_sem     = []   # semantic class per pair

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        codes_b    = codes_all[start:end].to(device)        # (B, L)
        acts_b     = actions_all[start:end].to(device)      # (B,)
        codes_b1   = codes_next_all[start:end].to(device)   # (B, L)
        sem_b      = sem_all[start:end]                      # (B, L) cpu

        with torch.no_grad():
            # WM takes flat discrete indices: (B, L) → need to embed
            # DiscreteTransitionModel.forward(x, acts) where x is (B, L) indices
            try:
                state_logits, _, _ = trans_model(codes_b, acts_b, return_logits=True)
                # state_logits: (B, n_codes, L)
                pred_codes = state_logits.argmax(dim=1)  # (B, L)
                correct = (pred_codes == codes_b1)       # (B, L) bool
            except Exception:
                # fallback: try forward_from_continuous path
                # this shouldn't happen for a discrete WM, skip
                continue

        all_correct.append(correct.cpu())
        all_sem.append(sem_b)

    if not all_correct:
        return {}

    all_correct = torch.cat(all_correct, dim=0).view(-1).float()  # (N*L,)
    all_sem     = torch.cat(all_sem,     dim=0).view(-1)           # (N*L,)

    per_class = {}
    for c in range(N_CLASSES):
        mask = all_sem == c
        count = mask.sum().item()
        if count < 10:
            continue
        acc = all_correct[mask].mean().item()
        per_class[c] = {'acc': acc, 'count': count, 'name': OBJECT_NAMES[c]}

    return per_class


# ── Semantic probe (lightweight, per-class) ───────────────────────────────

def compute_probe_accuracy_per_class(ae_model, env, n_frames, device, n_lat_side):
    """Quick logistic regression probe on quantized embeddings."""
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
            flat_emb = ae_model.encode(obs_in, return_quantized=True)  # (1, L*D)

        # Reshape to (L, D)
        n_lat = ae_model.n_latent_embeds
        emb_dim = flat_emb.shape[-1] // n_lat
        per_token = flat_emb.view(n_lat, emb_dim).cpu()

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
        sem_t = ge_rowmajor.flatten()

        emb_list.append(per_token.numpy())
        sem_list.append(sem_t)

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

    X = np.concatenate(emb_list, axis=0)   # (N*L, D)
    y = np.concatenate(sem_list, axis=0)   # (N*L,)

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
    parser.add_argument('--n_frames',   type=int, default=5000)
    parser.add_argument('--probe_frames', type=int, default=20000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()

    ae_model, trans_model, env, margs = load_checkpoint(args.model_path, args.device)
    n_lat = ae_model.n_latent_embeds
    n_lat_side = int(round(n_lat ** 0.5))

    # ── Semantic probe per class ──
    print(f"\nRunning semantic probe ({args.probe_frames} frames)...")
    env2 = make_env(margs.env_name)
    probe_acc = compute_probe_accuracy_per_class(
        ae_model, env2, args.probe_frames, args.device, n_lat_side)

    # ── WM accuracy per class ──
    wm_acc = {}
    if trans_model is not None:
        print(f"\nCollecting {args.n_frames} transitions for WM analysis...")
        codes, actions, codes_next, sem, flat_emb = collect_transitions(
            env, ae_model, trans_model, args.n_frames, args.device, n_lat_side)

        print("Computing WM per-class accuracy...")
        wm_acc = compute_wm_accuracy_per_class(
            trans_model, codes, actions, codes_next, sem, args.device)
    else:
        print("\n⚠ No transition model in checkpoint — skipping WM analysis.")

    # ── Report ──
    print("\n" + "=" * 70)
    print("PER-CLASS ANALYSIS")
    print("=" * 70)
    print(f"{'Class':<12} {'Probe Acc':>10} {'WM Acc':>10} {'Count':>8}")
    print("-" * 45)

    all_classes = sorted(set(list(probe_acc.keys()) + list(wm_acc.keys())))
    scatter_data = []
    for c in all_classes:
        p = probe_acc.get(c, {})
        w = wm_acc.get(c, {})
        p_acc  = p.get('acc', None)
        w_acc  = w.get('acc', None)
        count  = p.get('count', w.get('count', 0))
        name   = OBJECT_NAMES[c]
        p_str  = f"{p_acc*100:6.1f}%" if p_acc is not None else "     —"
        w_str  = f"{w_acc*100:6.1f}%" if w_acc is not None else "     —"
        print(f"  {name:<10} {p_str:>10} {w_str:>10} {count:>8,}")
        if p_acc is not None and w_acc is not None:
            scatter_data.append({
                'class': c, 'name': name,
                'probe_acc': round(p_acc, 4),
                'wm_acc':    round(w_acc, 4),
                'count':     int(count),
            })

    # Pearson correlation
    if len(scatter_data) >= 3:
        xs = np.array([d['probe_acc'] for d in scatter_data])
        ys = np.array([d['wm_acc']    for d in scatter_data])
        corr = float(np.corrcoef(xs, ys)[0, 1])
        print(f"\nPearson r (probe vs WM accuracy): {corr:.3f}")
    else:
        corr = None
        print("\n(Not enough classes for correlation)")

    # ── Save ──
    result = {
        'model_path':  args.model_path,
        'encoder_version': str(getattr(margs, 'ae_model_version', '?')),
        'env_name':    margs.env_name,
        'probe_acc_per_class': {
            str(c): {'acc': v['acc'], 'count': v['count'], 'name': v['name']}
            for c, v in probe_acc.items()
        },
        'wm_acc_per_class': {
            str(c): {'acc': v['acc'], 'count': v['count'], 'name': v['name']}
            for c, v in wm_acc.items()
        },
        'scatter_data': scatter_data,
        'pearson_r':   corr,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output_json}")

    return result


if __name__ == '__main__':
    main()
