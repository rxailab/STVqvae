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
from collections import Counter

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
                # Detect optional ConvNet architecture override (caveat-3 sweep)
                arch = getattr(margs, 'oracle_arch', 'mlp')
                if arch == 'spatial_conv':
                    sys.path.insert(0, os.path.dirname(__file__))
                    from train_conv_oracle_wm import SpatialConvTransitionModel
                    trans_model = SpatialConvTransitionModel(
                        encoder_out_shape=ae_model.encoder_out_shape,
                        n_actions=env.action_space.n,
                        hidden=getattr(margs, 'conv_hidden', 64),
                        depth=getattr(margs, 'conv_depth', 3),
                    )
                else:
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
            print(f"  WARN: could not load transition model: {e}")
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


# ── Extended (apples-to-apples) metrics: E1/E2/E3/E5/E6 ───────────────────
#
# E1  WM_probe       :  apply the SAME logistic probe (fit on z_0) to z_hat_k
# E2  WM_class (VQ)  :  code->class majority lookup; score class equality
# E3  WM_centroid    :  arg-max-cosine to per-class centroid (VAE positive control)
# E5  confusion      :  (true_class@0, probe_pred_class@k) 11x11 histogram
# E6  WM_swap (VAE)  :  relax exact metric to allow same-class spatial swaps
# All metrics bucket cells by ground-truth class at t=0 (paper §3.3a).


def fit_shared_probe(ae_model, env, n_frames, device, n_lat_side, trans_kind):
    """Returns (probe, X_NL_D, y_NL, codes_NL_or_None, per_class_recall_dict).

    A single shared logistic regression is fit once on per-token z_0 and reused
    for E1 (probe-the-WM-output) and for the original probe accuracy report.
    """
    from sklearn.linear_model import LogisticRegression

    ae_model.eval()
    emb_list, sem_list, code_list = [], [], []

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs_t = torch.from_numpy(obs).float()

    for _ in range(n_frames):
        obs_in = obs_t.unsqueeze(0).to(device)
        with torch.no_grad():
            try:
                flat_emb = ae_model.encode(obs_in, return_quantized=True)
            except TypeError:
                flat_emb = ae_model.encode(obs_in)

        latent_dim = flat_emb.shape[-1]
        L_tokens = n_lat_side * n_lat_side
        emb_dim = latent_dim // L_tokens
        emb_list.append(flat_emb.view(L_tokens, emb_dim).cpu().numpy())
        sem_list.append(semantic_grid(env, n_lat_side).numpy())

        if trans_kind == 'discrete':
            with torch.no_grad():
                codes = ae_model.encode(obs_in, return_quantized=False)
            if codes.dim() == 3:
                codes = codes.view(1, -1)
            code_list.append(codes.long().squeeze(0).cpu().numpy())

        a = env.action_space.sample()
        step_result = env.step(a)
        obs = step_result[0] if isinstance(step_result, tuple) else step_result
        done = step_result[2] if len(step_result) >= 3 else False
        obs_t = torch.from_numpy(obs).float()
        if done:
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs_t = torch.from_numpy(obs).float()

    X = np.concatenate(emb_list, axis=0)
    y = np.concatenate(sem_list, axis=0)
    codes = np.concatenate(code_list, axis=0) if code_list else None

    probe = LogisticRegression(max_iter=300, class_weight='balanced', C=1.0,
                               n_jobs=-1)
    probe.fit(X, y)
    y_pred = probe.predict(X)

    per_class = {}
    for c in np.unique(y):
        mask = y == c
        n = int(mask.sum())
        if n < 10:
            continue
        per_class[int(c)] = {
            'acc': float((y_pred[mask] == c).mean()),
            'count': n,
            'name': OBJECT_NAMES[int(c)],
        }
    return probe, X, y, codes, per_class


def build_code_to_class(codes, sem, n_codes):
    """Majority-vote class label for each code index. Returns (n_codes,) int."""
    lookup = -np.ones(n_codes, dtype=np.int64)
    for k in range(n_codes):
        m = codes == k
        if not m.any():
            continue
        lookup[k] = Counter(sem[m].tolist()).most_common(1)[0][0]
    return lookup


def build_class_centroids(X, y):
    """Per-class mean embedding from probe data. Returns (centroids, class_ids)."""
    classes = sorted(int(c) for c in np.unique(y))
    centroids, class_ids = [], []
    for c in classes:
        m = y == c
        if int(m.sum()) < 10:
            continue
        centroids.append(X[m].mean(axis=0))
        class_ids.append(c)
    return (np.stack(centroids).astype(np.float32),
            np.array(class_ids, dtype=np.int64))


def _bucket_per_class(correct_RL, sem0_RL):
    """correct/sem0 each (R, L). Returns {class_id: {acc, count, name}}."""
    correct = correct_RL.astype(np.float32).reshape(-1)
    classes = sem0_RL.reshape(-1)
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


def _score_probe_wm(probe, pred_emb_RLD, sem0_RL):
    """E1: apply shared probe to predicted continuous embedding; bucket by sem0."""
    R, L, D = pred_emb_RLD.shape
    pred_class = probe.predict(pred_emb_RLD.reshape(R * L, D)).reshape(R, L)
    correct = (pred_class == sem0_RL)
    return _bucket_per_class(correct, sem0_RL), pred_class


def _score_class_vq(pred_idx_RL, gt_idx_RL, code_to_class, sem0_RL):
    """E2: VQ class-equality via code->class lookup."""
    pred_cls = code_to_class[pred_idx_RL]
    gt_cls = code_to_class[gt_idx_RL]
    correct = (pred_cls == gt_cls) & (pred_cls >= 0)
    return _bucket_per_class(correct, sem0_RL)


def _score_centroid_vae(pred_emb_RLD, centroids_KD, class_ids_K, sem0_RL):
    """E3: arg-max cosine to class centroids, score class match against sem0."""
    R, L, D = pred_emb_RLD.shape
    z = torch.from_numpy(pred_emb_RLD)
    c = torch.from_numpy(centroids_KD)
    sims = torch.einsum('rld,kd->rlk',
                        F.normalize(z, dim=-1),
                        F.normalize(c, dim=-1))
    pred_class = class_ids_K[sims.argmax(dim=-1).numpy()]
    correct = (pred_class == sem0_RL)
    return _bucket_per_class(correct, sem0_RL)


def _score_swap_vae(pred_emb_RLD, gt_emb_RLD, sem_at_k_RL, sem0_RL):
    """E6: relax exact metric -- a predicted token is correct iff it is cosine-
    nearest to ANY ground-truth token in the same frame whose class equals
    the class at that prediction's position (i.e. same-class spatial swaps OK).
    """
    R, L, D = pred_emb_RLD.shape
    p = F.normalize(torch.from_numpy(pred_emb_RLD), dim=-1)
    g = F.normalize(torch.from_numpy(gt_emb_RLD), dim=-1)
    sims = torch.einsum('rld,rmd->rlm', p, g)            # (R, L_pred, L_true)
    nearest_pos = sims.argmax(dim=-1).numpy()             # (R, L)
    nearest_class = np.take_along_axis(sem_at_k_RL, nearest_pos, axis=1)
    correct = (nearest_class == sem_at_k_RL)
    return _bucket_per_class(correct, sem0_RL)


def _confusion_11x11(pred_RL, true_RL):
    """E5: (true@0, predicted@k) histogram, 11x11."""
    p = pred_RL.reshape(-1)
    t = true_RL.reshape(-1)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    valid = (t >= 0) & (t < N_CLASSES) & (p >= 0) & (p < N_CLASSES)
    np.add.at(cm, (t[valid], p[valid]), 1)
    return cm


def _class_pearson(probe_per_class, wm_per_class,
                   classes=('wall', 'door', 'key', 'goal', 'agent')):
    target_ids = [OBJECT_TO_IDX[n] for n in classes if n in OBJECT_TO_IDX]
    xs, ys = [], []
    for cid in target_ids:
        if cid in probe_per_class and cid in wm_per_class:
            xs.append(probe_per_class[cid]['acc'])
            ys.append(wm_per_class[cid]['acc'])
    if len(xs) < 3:
        return None
    return float(np.corrcoef(np.array(xs), np.array(ys))[0, 1])


def compute_all_multistep(ae_model, trans_model, trans_kind, env,
                          n_rollouts, horizons, device, n_lat_side, probe,
                          code_to_class=None, centroids=None,
                          centroid_class_ids=None):
    """Run rollouts ONCE; compute the original WM_exact metric AND the extended
    E1/E2/E3/E5/E6 metrics. Returns (original_per_h, extended_per_h).

    `original_per_h[k]` matches `compute_multistep_accuracy(...)[k]` exactly
    (per-class dict with acc/count/name).
    """
    max_steps = max(horizons)
    L = n_lat_side * n_lat_side
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L

    pred_emb = {k: [] for k in horizons}
    gt_emb = {k: [] for k in horizons}
    pred_idx = {k: [] for k in horizons}
    gt_idx = {k: [] for k in horizons}
    sem0 = {k: [] for k in horizons}
    sem_at_k = {k: [] for k in horizons}

    if trans_kind == 'discrete':
        try:
            cb_weight = ae_model.quantizer._embedding.weight.detach().cpu().numpy()
        except AttributeError as e:
            raise RuntimeError(
                f'Cannot find VQ codebook weights on ae_model.quantizer: {e}')

    for r in range(n_rollouts):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs_t = torch.from_numpy(obs).float()

        sem_seq = [semantic_grid(env, n_lat_side).numpy()]
        if trans_kind == 'discrete':
            codes_seq = [encode_codes(ae_model, obs_t, device)]
            with torch.no_grad():
                lat0 = ae_model.encode(obs_t.unsqueeze(0).to(device),
                                       return_quantized=True)
            lat_seq = [lat0.view(L, embedding_dim).cpu().numpy()]
        else:
            codes_seq = None
            lat_seq = [encode_latent(ae_model, obs_t, device)
                       .view(L, embedding_dim).cpu().numpy()]

        actions = []
        reached = 0
        for step in range(max_steps):
            a = env.action_space.sample()
            actions.append(a)
            step_result = env.step(a)
            obs_next = step_result[0] if isinstance(step_result, tuple) else step_result
            done = step_result[2] if len(step_result) >= 3 else False

            obs_next_t = torch.from_numpy(obs_next).float()
            sem_seq.append(semantic_grid(env, n_lat_side).numpy())
            if trans_kind == 'discrete':
                codes_seq.append(encode_codes(ae_model, obs_next_t, device))
                with torch.no_grad():
                    lat = ae_model.encode(obs_next_t.unsqueeze(0).to(device),
                                          return_quantized=True)
                lat_seq.append(lat.view(L, embedding_dim).cpu().numpy())
            else:
                lat_seq.append(encode_latent(ae_model, obs_next_t, device)
                               .view(L, embedding_dim).cpu().numpy())
            reached = step + 1
            if done:
                break

        if reached < 1:
            continue

        if trans_kind == 'discrete':
            preds = rollout_discrete(trans_model, codes_seq[0],
                                     actions[:reached], device)
        else:
            preds = rollout_continuous(trans_model, ae_model,
                                       torch.from_numpy(lat_seq[0]).reshape(-1).to(device),
                                       actions[:reached], device)

        for k in horizons:
            if k > reached:
                continue
            sem0[k].append(sem_seq[0])
            sem_at_k[k].append(sem_seq[k])
            gt_emb[k].append(lat_seq[k])
            if trans_kind == 'discrete':
                p_idx = preds[k - 1].numpy()
                pred_idx[k].append(p_idx)
                gt_idx[k].append(codes_seq[k].numpy())
                pred_emb[k].append(cb_weight[p_idx])
            else:
                pred_emb[k].append(preds[k - 1].view(L, embedding_dim).cpu().numpy())

        if (r + 1) % 50 == 0:
            print(f'  rollouts: {r + 1}/{n_rollouts}')

    original = {}
    extended = {}
    for k in horizons:
        if not pred_emb[k]:
            continue
        pe = np.stack(pred_emb[k])
        ge = np.stack(gt_emb[k])
        s0 = np.stack(sem0[k])
        sk = np.stack(sem_at_k[k])

        if trans_kind == 'discrete':
            pi = np.stack(pred_idx[k])
            gi = np.stack(gt_idx[k])
            correct_exact = (pi == gi)
        else:
            p_n = F.normalize(torch.from_numpy(pe), dim=-1)
            g_n = F.normalize(torch.from_numpy(ge), dim=-1)
            sims = torch.einsum('rld,rmd->rlm', p_n, g_n)
            correct_exact = (sims.argmax(dim=-1).numpy()
                             == np.arange(pe.shape[1])[None, :])
        original[k] = _bucket_per_class(correct_exact, s0)

        wm_probe, probe_pred = _score_probe_wm(probe, pe, s0)
        cm = _confusion_11x11(probe_pred, s0)

        ext = {
            'wm_probe_per_class': wm_probe,
            'confusion': cm.tolist(),
        }
        if trans_kind == 'discrete' and code_to_class is not None:
            ext['wm_class_per_class'] = _score_class_vq(pi, gi, code_to_class, s0)
        if trans_kind != 'discrete':
            if centroids is not None:
                ext['wm_centroid_per_class'] = _score_centroid_vae(
                    pe, centroids, centroid_class_ids, s0)
            ext['wm_swap_per_class'] = _score_swap_vae(pe, ge, sk, s0)

        extended[k] = ext

    return original, extended


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
    parser.add_argument('--no_extended', action='store_true',
                        help="Disable E1/E2/E3/E5/E6 extended metrics (legacy mode)")
    args = parser.parse_args()

    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(
        args.model_path, args.device)
    n_lat = getattr(ae_model, 'n_latent_embeds', None)
    if n_lat is None:
        n_lat = ae_model.latent_dim // getattr(margs, 'embedding_dim', 64)
    n_lat_side = int(round(n_lat ** 0.5))

    # ── Probe ──
    probe_acc = {}
    probe_obj = X_probe = y_probe = codes_buf = None
    code_to_class = centroids = centroid_class_ids = None
    extended_per_h = {}

    if not args.skip_probe:
        print(f"\nRunning semantic probe ({args.probe_frames} frames)...")
        env2 = make_env(margs.env_name)
        if args.no_extended:
            probe_acc = compute_probe_accuracy_per_class(
                ae_model, env2, args.probe_frames, args.device, n_lat_side)
        else:
            probe_obj, X_probe, y_probe, codes_buf, probe_acc = fit_shared_probe(
                ae_model, env2, args.probe_frames, args.device, n_lat_side, trans_kind)
            if trans_kind == 'discrete' and codes_buf is not None:
                n_codes = getattr(margs, 'codebook_size', 64)
                code_to_class = build_code_to_class(codes_buf, y_probe, n_codes)
            elif trans_kind != 'discrete':
                centroids, centroid_class_ids = build_class_centroids(X_probe, y_probe)

    # ── Multi-step WM ──
    wm_multi = {}
    if trans_model is not None:
        print(f"\nMulti-step WM rollouts: {args.n_rollouts} × up to {max(args.horizons)} steps")
        if args.no_extended or probe_obj is None:
            wm_multi = compute_multistep_accuracy(
                ae_model, trans_model, trans_kind, env,
                args.n_rollouts, args.horizons, args.device, n_lat_side)
        else:
            wm_multi, extended_per_h = compute_all_multistep(
                ae_model, trans_model, trans_kind, env,
                args.n_rollouts, args.horizons, args.device, n_lat_side,
                probe_obj, code_to_class=code_to_class,
                centroids=centroids, centroid_class_ids=centroid_class_ids)
            for k, ext in extended_per_h.items():
                pr = {'probe': _class_pearson(probe_acc, ext.get('wm_probe_per_class', {}))}
                if 'wm_class_per_class' in ext:
                    pr['class'] = _class_pearson(probe_acc, ext['wm_class_per_class'])
                if 'wm_centroid_per_class' in ext:
                    pr['centroid'] = _class_pearson(probe_acc, ext['wm_centroid_per_class'])
                if 'wm_swap_per_class' in ext:
                    pr['swap'] = _class_pearson(probe_acc, ext['wm_swap_per_class'])
                ext['pearson_r'] = pr
    else:
        print("\nWARN: no transition model in checkpoint -- skipping WM.")

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

    if extended_per_h:
        print("\nPearson r (probe vs WM_*  --  apples-to-apples extension):")
        variants = sorted({v for ext in extended_per_h.values()
                           for v in ext.get('pearson_r', {}).keys()})
        header = f"  {'k':>3}"
        for v in variants:
            header += f"  {v:>10}"
        print(header)
        for k in args.horizons:
            ext = extended_per_h.get(k)
            if not ext:
                continue
            line = f"  {k:>3}"
            pr = ext.get('pearson_r', {})
            for v in variants:
                val = pr.get(v)
                line += f"  {val:+10.3f}" if val is not None else f"  {'—':>10}"
            print(line)

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
    if extended_per_h:
        result['extended_metrics_per_horizon'] = {
            str(k): {
                **{name: ({str(c): v for c, v in d.items()} if isinstance(d, dict) else d)
                   for name, d in ext.items()
                   if name not in ('confusion', 'pearson_r')},
                'confusion': ext['confusion'],
                'pearson_r': ext.get('pearson_r', {}),
            }
            for k, ext in extended_per_h.items()
        }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output_json}")

    return result


if __name__ == '__main__':
    main()
