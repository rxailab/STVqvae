"""
Crafter semantic probe — Exp 44 / general Crafter VQVAE.

Loads ae_model from a direct checkpoint path (mf_e2e_semantic_crafter_*.pt),
rolls out a random policy, collects (codebook_index, semantic_label) pairs,
trains a logistic regression probe, and reports per-class accuracy.

Usage:
    python probe_crafter_semantics.py \
        --model_path models/crafter/mf_e2e_semantic_crafter_v6enc_fix_best_model.pt \
        --n_frames 30000 \
        --device cuda
"""

import argparse
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model


# ── Crafter semantic class names (crafter uses 0-based internal IDs) ──────────
# Source: crafter/constants.py  materials list
CRAFTER_CLASSES = {
    0:  'invalid',
    1:  'water',
    2:  'grass',
    3:  'stone',
    4:  'path',
    5:  'sand',
    6:  'tree',
    7:  'lava',
    8:  'coal',
    9:  'iron',
    10: 'diamond',
    11: 'table',
    12: 'furnace',
    13: 'plant',
    14: 'fence',
    15: 'player',
    16: 'cow',
    17: 'zombie',
    18: 'skeleton',
}
N_CLASSES = len(CRAFTER_CLASSES)  # 19


# ─────────────────────────────────────────────────────────────────────────────
def collect_data(env, ae_model, n_frames, device, n_lat_side):
    ae_model.eval()
    indices_list = []
    labels_list  = []

    obs, _ = env.reset()
    obs_t = preprocess_obs([obs]).to(device)

    with torch.no_grad():
        for step in range(n_frames):
            # Encode → get discrete codebook indices
            enc_out = ae_model.encode(obs_t, return_quantized=False)
            # enc_out shape: (1, n_latents) or (1, n_latents, K)
            if enc_out.ndim == 3:                       # one-hot (B,N,K)
                idx = enc_out.argmax(-1).squeeze(0)     # (N,)
            elif enc_out.ndim == 2:                     # already indices (B,N)
                idx = enc_out.squeeze(0)                # (N,)
            else:
                idx = enc_out.view(-1)

            # Semantic map from Crafter env
            inner = env
            while hasattr(inner, 'env'):
                inner = inner.env
            sem_map = None
            if hasattr(inner, 'get_semantic'):
                sem_map = inner.get_semantic()
            elif hasattr(inner, '_sem_view'):
                sem_map = inner._sem_view()

            if sem_map is None:
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    obs, _ = env.reset()
                obs_t = preprocess_obs([obs]).to(device)
                continue

            # Resize semantic map to match latent grid (n_lat_side × n_lat_side)
            from PIL import Image as _PIL
            sem_resized = np.array(
                _PIL.fromarray(sem_map.astype(np.uint8)).resize(
                    (n_lat_side, n_lat_side), resample=0),
                dtype=np.int64)

            indices_list.append(idx.cpu())
            labels_list.append(torch.tensor(sem_resized.flatten(), dtype=torch.long))

            # Step
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            obs_t = preprocess_obs([obs]).to(device)

            if (step + 1) % 5000 == 0:
                print(f'  {step+1}/{n_frames} frames collected')

    indices = torch.stack(indices_list, 0)   # (T, N)
    labels  = torch.stack(labels_list,  0)   # (T, N)
    return indices, labels


# ─────────────────────────────────────────────────────────────────────────────
def train_probe(indices, labels, codebook_size, n_latents, n_classes,
                epochs, batch_size, device):
    """
    Simple linear probe: embedding_lookup(index) -> class logits.
    We learn a (codebook_size, n_classes) weight matrix.
    """
    probe = nn.Linear(codebook_size, n_classes, bias=True).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Flatten (T*N,)
    flat_idx    = indices.view(-1)   # LongTensor
    flat_labels = labels.view(-1)

    # Filter out invalid class (0) — background / map boundary
    valid = flat_labels > 0
    flat_idx    = flat_idx[valid]
    flat_labels = flat_labels[valid]

    # One-hot encode indices  (T*N, codebook_size)
    oh = torch.zeros(len(flat_idx), codebook_size)
    oh.scatter_(1, flat_idx.unsqueeze(1), 1.0)

    # 80/20 split per class (manual, avoids stratified issues with rare classes)
    n = len(oh)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx = perm[:split]
    test_idx  = perm[split:]

    oh_train = oh[train_idx].to(device)
    y_train  = flat_labels[train_idx].to(device)
    oh_test  = oh[test_idx].to(device)
    y_test   = flat_labels[test_idx].to(device)

    print(f'\nProbe training: {len(oh_train)} train / {len(oh_test)} test samples')

    for epoch in range(epochs):
        probe.train()
        perm2 = torch.randperm(len(oh_train), device=device)
        total_loss = 0.0
        n_batches  = 0
        for i in range(0, len(oh_train), batch_size):
            b = perm2[i:i+batch_size]
            logits = probe(oh_train[b])
            loss   = loss_fn(logits, y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{epochs}  loss={total_loss/n_batches:.4f}')

    return probe, oh_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_probe(probe, oh_test, y_test, device):
    probe.eval()
    with torch.no_grad():
        preds = probe(oh_test).argmax(-1)

    overall_acc = (preds == y_test).float().mean().item()
    print(f'\nOverall accuracy: {overall_acc*100:.1f}%')

    present_classes = sorted(y_test.unique().tolist())
    print(f'\nPer-class accuracy (classes present in test set):')
    print(f'  {"Class":15s}  {"N":>7s}  {"Acc":>6s}')
    print('  ' + '-' * 35)
    per_class_accs = {}
    for c in present_classes:
        mask = y_test == c
        n    = mask.sum().item()
        acc  = (preds[mask] == c).float().mean().item()
        name = CRAFTER_CLASSES.get(c, f'class_{c}')
        marker = ' ✓' if acc > 0.1 else ''
        print(f'  {name:15s}  {n:7d}  {acc*100:5.1f}%{marker}')
        per_class_accs[c] = acc

    n_nonzero = sum(1 for a in per_class_accs.values() if a > 0.01)
    print(f'\nClasses with >1% accuracy: {n_nonzero} / {len(present_classes)}')
    return overall_acc, per_class_accs


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--n_frames',   type=int,   default=30000)
    parser.add_argument('--probe_epochs', type=int, default=30)
    parser.add_argument('--probe_batch_size', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f'Loading checkpoint: {args.model_path}')
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    saved_args_dict = ckpt['args']

    # Reconstruct minimal args namespace for construct_ae_model
    margs = types.SimpleNamespace(**saved_args_dict)
    margs.load = False       # we load weights manually below
    margs.e2e_loss = True
    margs.use_amp  = False
    margs.ae_recon_loss = getattr(margs, 'ae_recon_loss', False)
    margs.dead_code_threshold = getattr(margs, 'dead_code_threshold', 0.0)

    print(f'Env: {margs.env_name}')
    print(f'Codebook size: {margs.codebook_size}  |  Embedding dim: {margs.embedding_dim}')
    print(f'Filter size: {margs.filter_size}  |  Model version: {margs.ae_model_version}')
    print(f'Dead code threshold: {margs.dead_code_threshold}')

    env = make_env(margs.env_name)
    obs, _ = env.reset()
    sample_obs = preprocess_obs([obs])

    ae_model, _ = construct_ae_model(sample_obs.shape[1:], margs,
                                     latent_activation=True, load=False)
    ae_model.load_state_dict(ckpt['ae_model_state_dict'])
    ae_model = ae_model.to(args.device)
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad = False

    n_latents   = ae_model.n_latent_embeds
    n_lat_side  = int(round(n_latents ** 0.5))
    codebook_sz = margs.codebook_size
    print(f'\nLatent grid: {n_lat_side}×{n_lat_side} = {n_latents} tokens')

    # Check codebook usage from EMA buffers
    ema_usage = ae_model.quantizer._ema_cluster_size.detach().cpu()
    n_used = (ema_usage > 0.5).sum().item()
    print(f'Codebook utilisation (EMA>0.5): {n_used}/{codebook_sz} '
          f'({n_used/codebook_sz*100:.1f}%)')

    print(f'\nCollecting {args.n_frames} frames with random policy...')
    indices, labels = collect_data(env, ae_model, args.n_frames,
                                   args.device, n_lat_side)
    print(f'Collected: indices={tuple(indices.shape)}, labels={tuple(labels.shape)}')

    # Class distribution
    flat_labels = labels.view(-1)
    valid_labels = flat_labels[flat_labels > 0]
    print(f'\nLabel distribution ({len(valid_labels)} valid pixels):')
    for c in sorted(CRAFTER_CLASSES.keys()):
        count = (valid_labels == c).sum().item()
        if count > 0:
            name = CRAFTER_CLASSES[c]
            print(f'  {name:15s} (id={c:2d}): {count:7d} ({count/len(valid_labels)*100:.1f}%)')

    probe, oh_test, y_test = train_probe(
        indices, labels, codebook_sz, n_latents, N_CLASSES,
        args.probe_epochs, args.probe_batch_size, args.device)

    overall_acc, per_class_accs = evaluate_probe(probe, oh_test, y_test, args.device)

    print(f'\n{"="*50}')
    print(f'SUMMARY')
    print(f'  Model : {os.path.basename(args.model_path)}')
    print(f'  Codebook util: {n_used}/{codebook_sz} ({n_used/codebook_sz*100:.1f}%)')
    print(f'  Overall probe accuracy: {overall_acc*100:.1f}%')
    n_gt10 = sum(1 for a in per_class_accs.values() if a > 0.1)
    print(f'  Classes with >10% accuracy: {n_gt10} / {len(per_class_accs)}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
