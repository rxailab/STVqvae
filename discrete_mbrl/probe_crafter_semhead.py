"""
Evaluates the TRAINED semantic head (from checkpoint) on fresh Crafter rollout.
This tests whether the sem_aux loss actually taught the encoder+head to predict
semantic classes, rather than probing with a new untrained linear layer.

Usage:
    python probe_crafter_semhead.py \
        --model_path model_free/models/crafter/mf_e2e_semantic_crafter_v6enc_fix_best_model.pt \
        --n_frames 10000 --device cuda
"""

import argparse, os, sys, types, math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as _PIL

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model
from shared.models.transition_models import SemanticHeadV2

CRAFTER_CLASSES = {1:'water',2:'grass',3:'stone',4:'path',5:'sand',6:'tree',
                   7:'lava',8:'coal',9:'iron',10:'diamond',13:'plant',
                   14:'fence',15:'player',16:'cow',17:'zombie',18:'skeleton'}
N_CLASSES = 19


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_frames', type=int, default=10000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f'Loading: {args.model_path}')
    ckpt  = torch.load(args.model_path, map_location='cpu', weights_only=False)
    margs = types.SimpleNamespace(**ckpt['args'])
    margs.load = False; margs.e2e_loss = True; margs.use_amp = False
    margs.ae_recon_loss = getattr(margs, 'ae_recon_loss', False)
    margs.dead_code_threshold = getattr(margs, 'dead_code_threshold', 0.0)

    env = make_env(margs.env_name)
    obs, _ = env.reset()
    sample_obs = preprocess_obs([obs])

    ae_model, _ = construct_ae_model(sample_obs.shape[1:], margs,
                                     latent_activation=True, load=False)
    ae_model.load_state_dict(ckpt['ae_model_state_dict'])
    ae_model = ae_model.to(args.device).eval()
    for p in ae_model.parameters(): p.requires_grad = False

    n_latents  = ae_model.n_latent_embeds
    n_lat_side = int(round(n_latents ** 0.5))
    embed_dim  = margs.embedding_dim

    # Load trained semantic head
    sem_head = SemanticHeadV2(
        n_latent=n_latents,
        embedding_dim=embed_dim,
        n_classes=getattr(margs, 'sem_n_classes', N_CLASSES),
        hidden_dim=getattr(margs, 'sem_head_hidden', 128),
    )
    sem_head_sd = ckpt.get('sem_head_state_dict')
    if sem_head_sd is None:
        print('ERROR: No sem_head_state_dict in checkpoint!')
        return
    sem_head.load_state_dict(sem_head_sd)
    sem_head = sem_head.to(args.device).eval()
    for p in sem_head.parameters(): p.requires_grad = False

    print(f'Grid: {n_lat_side}x{n_lat_side}={n_latents} | embed_dim={embed_dim}')
    print(f'sem_pre_vq = {getattr(margs, "sem_pre_vq", False)}')

    # Evaluate on fresh rollout
    all_preds, all_labels = [], []
    obs_t = preprocess_obs([obs]).to(args.device)

    use_pre_vq = getattr(margs, 'sem_pre_vq', False)

    with torch.no_grad():
        for step in range(args.n_frames):
            if use_pre_vq:
                enc_out = ae_model.encoder(obs_t)            # pre-VQ (1,C,H,W)
                flat_feat = enc_out.reshape(1, -1)           # (1, C*N)
            else:
                # Quantized features
                enc_out = ae_model.encoder(obs_t)
                _, quantized, _, _ = ae_model.quantizer(enc_out)
                flat_feat = quantized.reshape(1, -1)         # (1, C*N)

            logits = sem_head(flat_feat)                     # (1, N, n_classes)
            preds  = logits.argmax(-1).squeeze(0).cpu()     # (N,)

            # Semantic map
            inner = env
            while hasattr(inner, 'env'): inner = inner.env
            sem_map = None
            if hasattr(inner, 'get_semantic'):   sem_map = inner.get_semantic()
            elif hasattr(inner, '_sem_view'):    sem_map = inner._sem_view()

            if sem_map is None:
                obs,_,t,tr,_ = env.step(env.action_space.sample())
                if t or tr: obs,_ = env.reset()
                obs_t = preprocess_obs([obs]).to(args.device)
                continue

            sem_r = np.array(_PIL.fromarray(sem_map.astype(np.uint8)).resize(
                (n_lat_side, n_lat_side), resample=0), dtype=np.int64)
            labels = torch.tensor(sem_r.flatten(), dtype=torch.long)

            all_preds.append(preds)
            all_labels.append(labels)

            action = env.action_space.sample()
            obs,_,t,tr,_ = env.step(action)
            if t or tr: obs,_ = env.reset()
            obs_t = preprocess_obs([obs]).to(args.device)
            if (step+1)%2000==0: print(f'  {step+1}/{args.n_frames} frames')

    all_preds  = torch.cat(all_preds,  0)
    all_labels = torch.cat(all_labels, 0)

    valid = all_labels > 0
    preds  = all_preds[valid]
    labels = all_labels[valid]

    overall = (preds == labels).float().mean().item()
    print(f'\n=== Trained Semantic Head Evaluation ===')
    print(f'Frames: {args.n_frames} | Valid tokens: {valid.sum().item():,}')
    print(f'sem_pre_vq: {use_pre_vq}')
    print(f'\nOverall accuracy: {overall*100:.1f}%')
    print(f'  {"Class":12s}  {"N":>7s}  {"Acc":>6s}  {"Pred%":>6s}')
    print('  ' + '-'*38)

    present = sorted(labels.unique().tolist())
    n_gt10 = 0
    for c in present:
        mask   = labels == c
        n      = mask.sum().item()
        acc_c  = (preds[mask] == c).float().mean().item()
        pred_c = (preds == c).sum().item() / len(preds) * 100
        if acc_c > 0.10: n_gt10 += 1
        name_c = CRAFTER_CLASSES.get(c, f'cls{c}')
        mark   = ' ✓' if acc_c > 0.10 else ''
        print(f'  {name_c:12s}  {n:7d}  {acc_c*100:5.1f}%  {pred_c:5.1f}%{mark}')

    print(f'\n  → Classes >10% accuracy: {n_gt10}/{len(present)}')

    # Prediction distribution
    print(f'\nPrediction distribution (what does the head actually predict?):')
    pred_counts = {}
    for c in preds.unique().tolist():
        pred_counts[c] = (preds == c).sum().item()
    for c, cnt in sorted(pred_counts.items(), key=lambda x: -x[1])[:10]:
        print(f'  class {c:2d} ({CRAFTER_CLASSES.get(c,"?"):10s}): {cnt:7d} ({cnt/len(preds)*100:.1f}%)')

    print(f'\n{"="*44}')
    print(f'SUMMARY: Trained sem_head accuracy = {overall*100:.1f}%')
    print(f'         Classes with >10% acc     = {n_gt10}/{len(present)}')


if __name__ == '__main__':
    main()
