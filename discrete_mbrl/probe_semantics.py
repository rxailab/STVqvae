"""
Phase 1 — Frozen semantic probe for Experiment 33.

Trains a small linear classifier on top of the frozen pretrained VQVAE to
measure how much object-type information the reconstruction-trained codebook
already encodes at each spatial position.

Usage:
    python probe_semantics.py \
        --env_name MiniGrid-LavaCrossingS9N1-v0 \
        --ae_model_hash ea136dc75d389f7b850959cd1f78eb6a \
        --n_frames 50000 \
        --probe_epochs 20 \
        --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs, OBJECT_TO_IDX
from model_construction import construct_ae_model, get_args
from training_helpers import update_params

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
N_CLASSES = 11  # 0..10

OBJECT_NAMES = [IDX_TO_OBJECT.get(i, f'class_{i}') for i in range(N_CLASSES)]


def collect_data(env, ae_model, n_frames, device, n_lat_side):
    """Roll out a random policy and collect (embedding, semantic_label) pairs."""
    ae_model.eval()
    embeddings = []
    sem_labels = []

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = torch.from_numpy(obs).float()

    collected = 0
    while collected < n_frames:
        with torch.no_grad():
            flat_emb = ae_model.encode(
                obs.unsqueeze(0).to(device), return_quantized=True)  # (1, n_latent*embed_dim)
        embeddings.append(flat_emb.cpu().squeeze(0))

        # Get semantic grid from unwrapped env
        ug = env.unwrapped
        ge = ug.grid.encode()[:, :, 0].copy()   # (width, height) col-major
        ge[ug.agent_pos[0], ug.agent_pos[1]] = OBJECT_TO_IDX['agent']
        ge_rowmajor = ge.T  # (height, width) row-major = image layout

        if ge_rowmajor.shape != (n_lat_side, n_lat_side):
            from PIL import Image as PIL_Image
            ge_rowmajor = np.array(
                PIL_Image.fromarray(ge_rowmajor.astype(np.uint8)).resize(
                    (n_lat_side, n_lat_side), resample=0),
                dtype=np.int64)

        sem_labels.append(torch.tensor(ge_rowmajor.flatten(), dtype=torch.long))  # (n_latent,)

        # Random action step
        act = env.action_space.sample()
        step_result = env.step(act)
        if len(step_result) == 5:
            next_obs, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_obs, _, done, _ = step_result

        if done:
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        else:
            obs = next_obs
        obs = torch.from_numpy(obs).float()

        collected += 1
        if collected % 5000 == 0:
            print(f'  Collected {collected}/{n_frames} frames')

    embeddings = torch.stack(embeddings)   # (N, n_latent * embed_dim)
    sem_labels = torch.stack(sem_labels)   # (N, n_latent)
    return embeddings, sem_labels


def train_probe(embeddings, sem_labels, embedding_dim, n_latent, n_classes,
                n_epochs, batch_size, device):
    """Train a shared linear probe: Linear(embedding_dim -> n_classes) per position."""
    probe = nn.Linear(embedding_dim, n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)

    N = embeddings.shape[0]
    # Reshape: (N, n_latent, embed_dim) and (N, n_latent)
    # VQVAE flat embeddings are channels-first: (N, C*n_latent) from (N, C, H, W).view(N,-1)
    emb_spatial = embeddings.view(N, embedding_dim, n_latent).permute(0, 2, 1)  # (N, n_latent, C)

    print(f'\nTraining linear probe: {embedding_dim} -> {n_classes} over {n_epochs} epochs')
    for epoch in range(n_epochs):
        idx = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            b_idx = idx[i:i + batch_size]
            b_emb = emb_spatial[b_idx].to(device)       # (B, n_latent, embed_dim)
            b_lbl = sem_labels[b_idx].to(device)        # (B, n_latent)
            B = b_emb.shape[0]

            logits = probe(b_emb)                       # (B, n_latent, n_classes)
            loss = F.cross_entropy(
                logits.view(B * n_latent, n_classes),
                b_lbl.view(B * n_latent),
                reduction='mean'
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{n_epochs}  loss={total_loss/n_batches:.4f}')

    return probe


def evaluate_probe(probe, embeddings, sem_labels, embedding_dim, n_latent, n_classes, device):
    """Compute overall and per-class accuracy."""
    probe.eval()
    N = embeddings.shape[0]
    # VQVAE flat embeddings are channels-first: (N, C*n_latent) from (N, C, H, W).view(N,-1)
    emb_spatial = embeddings.view(N, embedding_dim, n_latent).permute(0, 2, 1)  # (N, n_latent, C)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, N, 512):
            b_emb = emb_spatial[i:i+512].to(device)
            logits = probe(b_emb)                       # (B, n_latent, n_classes)
            preds = logits.argmax(dim=-1).cpu()         # (B, n_latent)
            all_preds.append(preds)
            all_targets.append(sem_labels[i:i+512])

    preds_flat = torch.cat(all_preds).view(-1)
    targets_flat = torch.cat(all_targets).view(-1)

    overall_acc = (preds_flat == targets_flat).float().mean().item()

    print(f'\n=== Probe Results ===')
    print(f'Overall per-position accuracy: {overall_acc*100:.1f}%')
    print(f'\nPer-class accuracy:')
    for c in range(n_classes):
        mask = targets_flat == c
        if mask.sum() == 0:
            continue
        acc = (preds_flat[mask] == targets_flat[mask]).float().mean().item()
        count = mask.sum().item()
        print(f'  {OBJECT_NAMES[c]:10s} (id={c}):  {acc*100:5.1f}%  ({count} samples)')

    # Confusion matrix
    print(f'\nConfusion matrix (rows=true, cols=pred):')
    present_classes = sorted(targets_flat.unique().tolist())
    header = '           ' + ''.join(f'{OBJECT_NAMES[c][:6]:>8}' for c in present_classes)
    print(header)
    for true_c in present_classes:
        mask = targets_flat == true_c
        row_preds = preds_flat[mask]
        row = f'{OBJECT_NAMES[true_c][:10]:10s} '
        for pred_c in present_classes:
            n = (row_preds == pred_c).sum().item()
            row += f'{n:8d}'
        print(row)

    return overall_acc


def main():
    parser = argparse.ArgumentParser(description='Phase 1 semantic probe for VQVAE')
    parser.add_argument('--env_name', default='MiniGrid-LavaCrossingS9N1-v0')
    parser.add_argument('--ae_model_hash', default='ea136dc75d389f7b850959cd1f78eb6a')
    parser.add_argument('--ae_model_type', default='vqvae')
    parser.add_argument('--ae_model_version', type=int, default=2)
    parser.add_argument('--codebook_size', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--filter_size', type=int, default=9)
    parser.add_argument('--n_frames', type=int, default=50000)
    parser.add_argument('--probe_epochs', type=int, default=20)
    parser.add_argument('--probe_batch_size', type=int, default=256)
    parser.add_argument('--model_dir', default='..')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Minimal extra attrs needed by construct_ae_model
    args.load = True
    args.e2e_loss = True   # so encode(return_quantized=True) works
    args.use_amp = False
    args.ae_recon_loss = False

    print(f'Environment : {args.env_name}')
    print(f'VQVAE hash  : {args.ae_model_hash}')
    print(f'Device      : {args.device}')

    env = make_env(args.env_name)
    reset_result = env.reset()
    sample_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([sample_obs])

    ae_model, _ = construct_ae_model(sample_obs.shape[1:], args,
                                     latent_activation=True, load=True)
    ae_model = ae_model.to(args.device)
    for p in ae_model.parameters():
        p.requires_grad = False
    ae_model.eval()

    n_latent = ae_model.n_latent_embeds
    n_lat_side = int(round(np.sqrt(n_latent)))
    print(f'VQVAE: n_latent={n_latent} ({n_lat_side}×{n_lat_side}), embed_dim={args.embedding_dim}')

    print(f'\nCollecting {args.n_frames} frames with random policy...')
    embeddings, sem_labels = collect_data(
        env, ae_model, args.n_frames, args.device, n_lat_side)
    print(f'Collected: embeddings={tuple(embeddings.shape)}, labels={tuple(sem_labels.shape)}')

    # Class distribution
    flat_labels = sem_labels.view(-1)
    print('\nLabel distribution:')
    for c in range(N_CLASSES):
        count = (flat_labels == c).sum().item()
        if count > 0:
            print(f'  {OBJECT_NAMES[c]:10s} (id={c}): {count} ({count/len(flat_labels)*100:.1f}%)')

    probe = train_probe(
        embeddings, sem_labels, args.embedding_dim, n_latent, N_CLASSES,
        args.probe_epochs, args.probe_batch_size, args.device)

    overall_acc = evaluate_probe(
        probe, embeddings, sem_labels, args.embedding_dim, n_latent, N_CLASSES, args.device)

    # Save probe
    save_path = f'../models/{args.env_name}/semantic_probe_{args.ae_model_hash[:8]}.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'probe_state_dict': probe.state_dict(),
        'overall_acc': overall_acc,
        'args': vars(args),
    }, save_path)
    print(f'\nProbe saved to: {save_path}')


if __name__ == '__main__':
    main()
