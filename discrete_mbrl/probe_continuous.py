"""
Semantic probe for continuous encoder (VAE / AE baseline).

Unlike probe_semantics.py which probes discrete VQ code indices, this script
probes the raw continuous encoder output (B, C, H, W) reshaped to per-position
C-dim feature vectors.  This lets us ask: does the continuous representation
naturally separate semantic classes even without a discrete bottleneck?

Usage:
    python probe_continuous.py \
        --model_path ../discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/vae_baseline_doorkey_v6enc_best_model.pt \
        --n_frames 30000 --device cuda \
        --output_json /tmp/vae_probe.json
"""

import argparse, os, sys, json
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs, OBJECT_TO_IDX

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}


# ── load model ───────────────────────────────────────────────────────────────

def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    margs = ckpt['args']
    if isinstance(margs, dict):
        class A:
            pass
        a = A()
        a.__dict__.update(margs)
        margs = a

    ae_type    = getattr(margs, 'ae_model_type', 'vae')
    ae_version = str(getattr(margs, 'ae_model_version', '6'))
    emb_dim    = getattr(margs, 'embedding_dim', 64)
    filt_size  = getattr(margs, 'filter_size', 8)
    env_name   = getattr(margs, 'env_name', 'MiniGrid-DoorKey-8x8-v0')

    # Build a sample obs to get input_dim
    env = make_env(env_name)
    obs, _ = env.reset()
    sample = preprocess_obs([obs])   # (1, C, H, W)
    input_dim = sample.shape[1:]
    env.close()

    from model_construction import make_ae
    encoder, decoder = make_ae(input_dim, emb_dim, filt_size, version=ae_version)

    if ae_type == 'vqvae':
        from shared.models.encoder_models import VQVAEModel
        codebook_size = getattr(margs, 'codebook_size', 64)
        ae_model = VQVAEModel(
            input_dim,
            codebook_size=codebook_size,
            embedding_dim=emb_dim,
            encoder=encoder,
            decoder=decoder,
            commitment_cost=0.25,
            ema_decay=0.99,
            dead_code_threshold=getattr(margs, 'dead_code_threshold', 0.0),
        )
    else:
        from shared.models.encoder_models import AEModel
        stochastic = (ae_type == 'vae')
        ae_model = AEModel(
            obs_dim=input_dim,
            latent_dim=None,   # keep spatial encoder output, no fc projection
            encoder=encoder,
            decoder=decoder,
            stochastic=stochastic,
        )

    ae_model.load_state_dict(ckpt['ae_model_state_dict'])
    ae_model.eval().to(device)

    return ae_model, margs, env_name


# ── data collection ──────────────────────────────────────────────────────────

def collect_data(env_name, ae_model, n_frames, device):
    """
    Returns:
        feats  : (N, C)  float32 — per-position continuous encoder feature
        labels : (N,)    int64   — semantic class id
    where N = n_frames * n_lat (number of spatial positions collected).
    """
    env = make_env(env_name)
    obs, _ = env.reset()

    # Determine spatial layout from one forward pass
    with torch.no_grad():
        x = preprocess_obs([obs]).to(device)
        enc_raw = ae_model.encoder(x)   # (1, C, H, W)
        _, C, H, W = enc_raw.shape
    n_lat = H * W
    n_lat_side = H

    feats_list  = []
    labels_list = []

    is_crafter = 'crafter' in env_name.lower()

    for step in range(n_frames):
        with torch.no_grad():
            x = preprocess_obs([obs]).to(device)
            enc_raw = ae_model.encoder(x)   # (1, C, H, W)
            # per-position features: (H*W, C)
            pos_feats = enc_raw[0].permute(1, 2, 0).reshape(n_lat, C).cpu().numpy()

        # semantic labels aligned to (H, W) grid
        ug = env.unwrapped
        if is_crafter:
            sem_map = np.zeros((H, W), dtype=np.int64)
        else:
            ge = ug.grid.encode()[:, :, 0].copy()
            ge[ug.agent_pos[0], ug.agent_pos[1]] = OBJECT_TO_IDX['agent']
            ge_row = ge.T   # (height, width) row-major
            if ge_row.shape != (n_lat_side, n_lat_side):
                from PIL import Image as _PIL
                ge_row = np.array(
                    _PIL.fromarray(ge_row.astype(np.uint8)).resize(
                        (n_lat_side, n_lat_side), resample=0),
                    dtype=np.int64)
            sem_map = ge_row

        labels = sem_map.flatten()   # (n_lat,)
        feats_list.append(pos_feats)
        labels_list.append(labels)

        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

        if (step + 1) % 5000 == 0:
            print(f'  {step+1}/{n_frames} frames')

    env.close()
    feats  = np.concatenate(feats_list,  axis=0).astype(np.float32)   # (N*n_lat, C)
    labels = np.concatenate(labels_list, axis=0).astype(np.int64)     # (N*n_lat,)
    return feats, labels


# ── probe ────────────────────────────────────────────────────────────────────

def run_probe(feats, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Only use classes that appear in data
    classes, counts = np.unique(labels, return_counts=True)
    print(f'\nClass distribution (total {len(labels):,} samples):')
    for cls, cnt in zip(classes, counts):
        name = IDX_TO_OBJECT.get(int(cls), f'cls{cls}')
        print(f'  {name:<12} (id={cls}): {cnt:>8,}  ({cnt/len(labels)*100:.1f}%)')

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    y = labels

    # Balanced logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
        n_jobs=-1,
    )
    clf.fit(X, y)
    preds = clf.predict(X)

    # Per-class accuracy
    print(f'\n{"="*60}')
    print('PER-CLASS PROBE ACCURACY (continuous latent)')
    print(f'{"="*60}')
    print(f'{"Class":<14} {"Accuracy":>10}  {"Count":>10}')
    print('-' * 40)
    results = {}
    overall_correct = 0
    for cls in classes:
        mask = labels == cls
        acc = (preds[mask] == cls).mean()
        name = IDX_TO_OBJECT.get(int(cls), f'cls{cls}')
        print(f'  {name:<12}  {acc*100:>8.1f}%  {mask.sum():>10,}')
        results[name] = {'acc': float(acc), 'count': int(mask.sum()), 'class_id': int(cls)}
        overall_correct += (preds[mask] == cls).sum()

    overall = overall_correct / len(labels)
    print(f'\n  Overall (macro-mean): {np.mean([v["acc"] for v in results.values()])*100:.1f}%')
    print(f'  Overall (weighted):   {overall*100:.1f}%')
    return results, clf, scaler


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_frames',   type=int, default=30000)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f'Loading: {args.model_path}')
    ae_model, margs, env_name = load_model(args.model_path, device)
    print(f'Env: {env_name}')
    print(f'Encoder type: {getattr(margs,"ae_model_type","?")}  version={getattr(margs,"ae_model_version","?")}')
    print(f'Latent dim: {ae_model.latent_dim}')

    print(f'\nCollecting {args.n_frames} frames...')
    feats, labels = collect_data(env_name, ae_model, args.n_frames, device)
    print(f'Feature matrix: {feats.shape}  Labels: {labels.shape}')

    results, clf, scaler = run_probe(feats, labels)

    if args.output_json:
        out = {
            'model_path': args.model_path,
            'env_name': env_name,
            'ae_model_type': getattr(margs, 'ae_model_type', '?'),
            'ae_model_version': str(getattr(margs, 'ae_model_version', '?')),
            'n_frames': args.n_frames,
            'per_class': results,
            'macro_mean_acc': float(np.mean([v['acc'] for v in results.values()])),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSaved to {args.output_json}')
