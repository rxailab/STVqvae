"""
Deep Crafter semantic probe — tests three representation levels:
  1. One-hot codebook indices  (1024-dim sparse)
  2. Quantized embedding vectors (64-dim dense)
  3. Pre-VQ encoder features (64-dim dense)

Usage:
    python probe_crafter_deep.py \
        --model_path model_free/models/crafter/mf_e2e_semantic_crafter_v6enc_fix_best_model.pt \
        --n_frames 20000 --device cuda
"""

import argparse, os, sys, types
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as _PIL

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model

CRAFTER_CLASSES = {1:'water',2:'grass',3:'stone',4:'path',5:'sand',6:'tree',
                   7:'lava',8:'coal',9:'iron',10:'diamond',13:'plant',
                   14:'fence',15:'player',16:'cow',17:'zombie',18:'skeleton'}
N_CLASSES = 19


# ─────────────────────────────────────────────────────────────────────────────
def collect_data(env, ae_model, n_frames, device, n_lat_side, embed_dim):
    ae_model.eval()
    idx_list, emb_list, pre_list, lbl_list = [], [], [], []
    obs, _ = env.reset()
    obs_t = preprocess_obs([obs]).to(device)

    with torch.no_grad():
        for step in range(n_frames):
            # Pre-VQ encoder features
            enc_out = ae_model.encoder(obs_t)                        # (1,C,H,W)
            _, quantized, _, oh_enc = ae_model.quantizer(enc_out)    # quantized (1,C,H,W)
            idx = oh_enc.argmax(dim=1).squeeze(0).cpu()              # (N,) long

            pre_flat   = enc_out.squeeze(0).view(embed_dim, -1).T.cpu()   # (N,C)
            quant_flat = quantized.squeeze(0).view(embed_dim, -1).T.cpu() # (N,C)

            # Semantic map
            inner = env
            while hasattr(inner, 'env'): inner = inner.env
            sem_map = None
            if hasattr(inner, 'get_semantic'):   sem_map = inner.get_semantic()
            elif hasattr(inner, '_sem_view'):    sem_map = inner._sem_view()

            if sem_map is None:
                obs,_,terminated,truncated,_ = env.step(env.action_space.sample())
                if terminated or truncated: obs,_ = env.reset()
                obs_t = preprocess_obs([obs]).to(device)
                continue

            sem_r = np.array(_PIL.fromarray(sem_map.astype(np.uint8)).resize(
                (n_lat_side, n_lat_side), resample=0), dtype=np.int64)

            idx_list.append(idx)
            emb_list.append(quant_flat)
            pre_list.append(pre_flat)
            lbl_list.append(torch.tensor(sem_r.flatten(), dtype=torch.long))

            action = env.action_space.sample()
            obs,_,terminated,truncated,_ = env.step(action)
            if terminated or truncated: obs,_ = env.reset()
            obs_t = preprocess_obs([obs]).to(device)
            if (step+1) % 5000 == 0:
                print(f'  {step+1}/{n_frames} frames collected')

    return (torch.stack(idx_list, 0).view(-1),
            torch.cat(emb_list, 0),
            torch.cat(pre_list, 0),
            torch.stack(lbl_list, 0).view(-1))


# ─────────────────────────────────────────────────────────────────────────────
def run_probe(name, X, Y, n_out, device, epochs=20, bs=1024):
    n_in = X.shape[1]
    perm  = torch.randperm(len(X))
    split = int(0.8 * len(X))
    Xtr = X[perm[:split]].to(device)
    Ytr = Y[perm[:split]].to(device)
    Xte = X[perm[split:]].to(device)
    Yte = Y[perm[split:]].to(device)

    probe   = nn.Linear(n_in, n_out).to(device)
    opt     = torch.optim.Adam(probe.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        probe.train()
        perm2 = torch.randperm(len(Xtr), device=device)
        total_loss = 0.0
        for i in range(0, len(Xtr), bs):
            b = perm2[i:i+bs]
            loss = loss_fn(probe(Xtr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if (ep+1) % 5 == 0:
            probe.eval()
            with torch.no_grad():
                preds = probe(Xte).argmax(-1)
                acc   = (preds == Yte).float().mean().item()
            print(f'  [{name}] Epoch {ep+1}/{epochs}  '
                  f'loss={total_loss/(len(Xtr)//bs+1):.4f}  test_acc={acc*100:.1f}%')

    probe.eval()
    with torch.no_grad():
        preds = probe(Xte).argmax(-1)

    overall = (preds == Yte).float().mean().item()
    present = sorted(Yte.unique().tolist())

    print(f'\n[{name}] Overall accuracy: {overall*100:.1f}%')
    print(f'  {"Class":12s}  {"N":>7s}  {"Acc":>6s}')
    print('  ' + '-'*32)
    n_gt10 = 0
    for c in present:
        mask  = Yte == c
        n     = mask.sum().item()
        acc_c = (preds[mask] == c).float().mean().item()
        if acc_c > 0.10: n_gt10 += 1
        name_c = CRAFTER_CLASSES.get(c, f'cls{c}')
        mark   = ' ✓' if acc_c > 0.10 else ''
        print(f'  {name_c:12s}  {n:7d}  {acc_c*100:5.1f}%{mark}')
    print(f'  → Classes >10%: {n_gt10}/{len(present)}')
    return overall


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_frames',      type=int, default=20000)
    parser.add_argument('--probe_epochs',  type=int, default=20)
    parser.add_argument('--probe_bs',      type=int, default=1024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f'Loading: {args.model_path}')
    ckpt  = torch.load(args.model_path, map_location='cpu', weights_only=False)
    margs = types.SimpleNamespace(**ckpt['args'])
    margs.load = False; margs.e2e_loss = True; margs.use_amp = False
    margs.ae_recon_loss      = getattr(margs, 'ae_recon_loss', False)
    margs.dead_code_threshold = getattr(margs, 'dead_code_threshold', 0.0)

    env = make_env(margs.env_name)
    obs, _ = env.reset()
    sample_obs = preprocess_obs([obs])

    ae_model, _ = construct_ae_model(sample_obs.shape[1:], margs,
                                     latent_activation=True, load=False)
    ae_model.load_state_dict(ckpt['ae_model_state_dict'])
    ae_model = ae_model.to(args.device).eval()
    for p in ae_model.parameters(): p.requires_grad = False

    n_latents   = ae_model.n_latent_embeds
    n_lat_side  = int(round(n_latents ** 0.5))
    codebook_sz = margs.codebook_size
    embed_dim   = margs.embedding_dim

    print(f'Grid: {n_lat_side}x{n_lat_side}={n_latents} tokens | '
          f'codebook={codebook_sz} | embed_dim={embed_dim}')
    ema_usage = ae_model.quantizer._ema_cluster_size.detach().cpu()
    n_used = (ema_usage > 0.5).sum().item()
    print(f'Codebook utilisation (EMA>0.5): {n_used}/{codebook_sz} '
          f'({n_used/codebook_sz*100:.0f}%)')

    print(f'\nCollecting {args.n_frames} frames...')
    all_idx, all_emb, all_pre, all_lbl = collect_data(
        env, ae_model, args.n_frames, args.device, n_lat_side, embed_dim)

    # Filter invalid (class 0)
    valid = all_lbl > 0
    all_idx, all_emb, all_pre, all_lbl = (
        all_idx[valid], all_emb[valid], all_pre[valid], all_lbl[valid])
    print(f'Valid tokens: {len(all_lbl):,}')

    print('\nLabel distribution:')
    for c in sorted(CRAFTER_CLASSES.keys()):
        n = (all_lbl == c).sum().item()
        if n > 0:
            print(f'  {CRAFTER_CLASSES[c]:12s}: {n:7d} ({n/len(all_lbl)*100:.1f}%)')

    # ── PROBE 1: one-hot codebook indices ─────────────────────────────────────
    print('\n' + '='*55)
    print('PROBE 1: One-hot codebook indices (1024-dim)')
    oh = torch.zeros(len(all_idx), codebook_sz)
    oh.scatter_(1, all_idx.unsqueeze(1).long(), 1.0)
    acc1 = run_probe('OneHot', oh, all_lbl, N_CLASSES, args.device,
                     args.probe_epochs, args.probe_bs)

    # ── PROBE 2: quantized embedding vectors ──────────────────────────────────
    print('\n' + '='*55)
    print('PROBE 2: Quantized embedding vectors (64-dim)')
    acc2 = run_probe('Quant', all_emb.float(), all_lbl, N_CLASSES, args.device,
                     args.probe_epochs, args.probe_bs)

    # ── PROBE 3: pre-VQ encoder features ─────────────────────────────────────
    print('\n' + '='*55)
    print('PROBE 3: Pre-VQ encoder features (64-dim)')
    acc3 = run_probe('PreVQ', all_pre.float(), all_lbl, N_CLASSES, args.device,
                     args.probe_epochs, args.probe_bs)

    print('\n' + '='*55)
    print('SUMMARY')
    print(f'  Codebook utilisation  : {n_used}/{codebook_sz} ({n_used/codebook_sz*100:.0f}%)')
    print(f'  Probe 1 (one-hot idx) : {acc1*100:.1f}%')
    print(f'  Probe 2 (quant emb)   : {acc2*100:.1f}%')
    print(f'  Probe 3 (pre-VQ feat) : {acc3*100:.1f}%')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
