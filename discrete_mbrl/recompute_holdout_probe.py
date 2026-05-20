#!/usr/bin/env python3
"""
Recompute per-class probe recall with a held-out train/test split.

Mirrors the probe protocol in analyze_wm_multistep.py::fit_shared_probe but
collects 2*n_frames frames, splits 50/50 train/test by random index, fits the
multinomial logistic regression on train only, and reports per-class recall on
the held-out test split.

Output: JSON with {class_id: {train_acc, test_acc, train_count, test_count, name}}
plus the per-class avg probe recall on each split (mean over classes with
sufficient support, matching the pooling used in §4.2).

Usage: python recompute_holdout_probe.py --model_path PATH --output_json OUT
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from analyze_wm_multistep import load_checkpoint, semantic_grid, OBJECT_NAMES
from env_helpers import make_env


def collect_embeddings(ae_model, env, n_frames, n_lat_side, device):
    """Replays the data-collection loop from fit_shared_probe but separates
    the train and test sweeps so they are sampled with independent rollouts.

    We collect 2*n_frames total: first n_frames go to train, last n_frames go
    to test. Episodes reset across the boundary, so the splits are temporally
    disjoint and the held-out evaluation is genuine.
    """
    ae_model.eval()
    emb_train, sem_train = [], []
    emb_test,  sem_test  = [], []

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs_t = torch.from_numpy(obs).float()

    for step in range(2 * n_frames):
        obs_in = obs_t.unsqueeze(0).to(device)
        with torch.no_grad():
            try:
                flat_emb = ae_model.encode(obs_in, return_quantized=True)
            except TypeError:
                flat_emb = ae_model.encode(obs_in)

        latent_dim = flat_emb.shape[-1]
        L_tokens = n_lat_side * n_lat_side
        emb_dim = latent_dim // L_tokens
        emb = flat_emb.view(L_tokens, emb_dim).cpu().numpy()
        sem = semantic_grid(env, n_lat_side).numpy()

        if step < n_frames:
            emb_train.append(emb); sem_train.append(sem)
        else:
            emb_test.append(emb); sem_test.append(sem)

        a = env.action_space.sample()
        step_result = env.step(a)
        obs = step_result[0] if isinstance(step_result, tuple) else step_result
        done = step_result[2] if len(step_result) >= 3 else False
        obs_t = torch.from_numpy(obs).float()
        if done:
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs_t = torch.from_numpy(obs).float()

    X_train = np.concatenate(emb_train, axis=0)
    y_train = np.concatenate(sem_train, axis=0)
    X_test  = np.concatenate(emb_test,  axis=0)
    y_test  = np.concatenate(sem_test,  axis=0)
    return X_train, y_train, X_test, y_test


def fit_and_score(X_tr, y_tr, X_te, y_te):
    """Fit a multinomial LR on train; report per-class recall on both splits."""
    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(
        max_iter=300, class_weight='balanced', C=1.0, n_jobs=-1)
    probe.fit(X_tr, y_tr)

    y_tr_pred = probe.predict(X_tr)
    y_te_pred = probe.predict(X_te)

    classes = sorted(set(np.unique(y_tr).tolist()) | set(np.unique(y_te).tolist()))
    per_class = {}
    for c in classes:
        m_tr = y_tr == c
        m_te = y_te == c
        n_tr = int(m_tr.sum())
        n_te = int(m_te.sum())
        if n_te < 10 or n_tr < 10:
            continue
        per_class[int(c)] = {
            'name': OBJECT_NAMES[int(c)],
            'train_count': n_tr,
            'test_count':  n_te,
            'train_acc': float((y_tr_pred[m_tr] == c).mean()),
            'test_acc':  float((y_te_pred[m_te] == c).mean()),
        }
    return per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--n_frames',   type=int, default=20000,
                    help='Frames per split. Total collection = 2*n_frames.')
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--output_json', required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ae_model, _trans, trans_kind, env, margs = load_checkpoint(args.model_path, device)

    # Determine n_lat_side from the encoder (matches analyze_wm_multistep usage).
    sample_reset = env.reset()
    sample_obs = sample_reset[0] if isinstance(sample_reset, tuple) else sample_reset
    sample_t = torch.from_numpy(sample_obs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        try:
            test_emb = ae_model.encode(sample_t, return_quantized=True)
        except TypeError:
            test_emb = ae_model.encode(sample_t)
    n_lat_side = int(np.sqrt(test_emb.shape[-1] // 64))
    if n_lat_side ** 2 * 64 != test_emb.shape[-1]:
        # Fall back to inferring side from margs.
        n_lat_side = int(getattr(margs, 'filter_size', 8))
    print(f"n_lat_side={n_lat_side}, embedding shape={tuple(test_emb.shape)}")

    X_tr, y_tr, X_te, y_te = collect_embeddings(
        ae_model, env, args.n_frames, n_lat_side, device)
    print(f"Collected: train X={X_tr.shape} y={y_tr.shape} | "
          f"test X={X_te.shape} y={y_te.shape}")

    per_class = fit_and_score(X_tr, y_tr, X_te, y_te)
    train_acc_avg = float(np.mean([v['train_acc'] for v in per_class.values()]))
    test_acc_avg  = float(np.mean([v['test_acc']  for v in per_class.values()]))
    delta = test_acc_avg - train_acc_avg

    out = {
        'model_path': args.model_path,
        'env_name':   getattr(margs, 'env_name', None),
        'n_frames_per_split': args.n_frames,
        'per_class': per_class,
        'probe_avg_recall_train': train_acc_avg,
        'probe_avg_recall_test':  test_acc_avg,
        'generalization_gap':     delta,
    }

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults -> {args.output_json}")
    print(f"  probe_avg_recall_train = {train_acc_avg:.4f}")
    print(f"  probe_avg_recall_test  = {test_acc_avg:.4f}")
    print(f"  generalization_gap     = {delta:+.4f}")


if __name__ == '__main__':
    main()
