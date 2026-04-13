"""
Comparative semantic probe: exp31 (wm-only) vs exp33 (wm + sem aux).

Loads the frozen encoder from each checkpoint, trains a shared linear probe
(nn.Linear(64, 11)) on quantized embeddings, and reports per-class accuracy.
Also visualises the predicted vs ground-truth object-type grid for a sample frame.

Usage (run from discrete_mbrl/):
    python eval_semantic_compare.py
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs, OBJECT_TO_IDX
from model_construction import construct_ae_model

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
N_CLASSES     = 11
OBJECT_NAMES  = [IDX_TO_OBJECT.get(i, f'cls{i}') for i in range(N_CLASSES)]

# ── checkpoints ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_free', 'models',
                         'MiniGrid-LavaCrossingS9N1-v0')
DK_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_free', 'models',
                            'MiniGrid-DoorKey-8x8-v0')
CHECKPOINTS = {
    'exp32_dk_wm_only':    os.path.join(DK_MODEL_DIR, 'mf_e2e_wm_doorkey_v1_best_model.pt'),
    'exp36_dk_v5enc':      os.path.join(DK_MODEL_DIR, 'mf_e2e_semantic_doorkey_v5enc_best_model.pt'),
    'exp37_dk_v5_prevq':   os.path.join(DK_MODEL_DIR, 'mf_e2e_semantic_doorkey_v5enc_prevq_best_model.pt'),
    'exp38_dk_v6_goal':    os.path.join(DK_MODEL_DIR, 'mf_e2e_semantic_doorkey_v6enc_goal_best_model.pt'),
    'exp39_dk_v7_multiscale': os.path.join(DK_MODEL_DIR, 'mf_e2e_semantic_doorkey_v7enc_multiscale_best_model.pt'),
    'exp40_dk_v8_gated': os.path.join(DK_MODEL_DIR, 'mf_e2e_semantic_doorkey_v8enc_gated_best_model.pt'),
    'exp41_dk_v9_patch': os.path.join(DK_MODEL_DIR, 'mf_e2e_semantic_doorkey_v9enc_patch_best_model.pt'),
}

DK16_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_free', 'models',
                              'MiniGrid-DoorKey-16x16-v0')
CHECKPOINTS_DK16 = {
    'exp42_dk16_v6_restart': os.path.join(DK16_MODEL_DIR, 'mf_e2e_semantic_doorkey16_v6enc_restart_best_model.pt'),
}

ENV_NAME     = 'MiniGrid-DoorKey-8x8-v0'
N_FRAMES     = 30_000   # frames per model
PROBE_EPOCHS = 20
BATCH        = 256
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

EMB_DIM = 64
# N_LAT and N_SIDE are now auto-detected per model from ae_model.n_latent_embeds

# ── helpers ───────────────────────────────────────────────────────────────────

def load_ae_from_checkpoint(ckpt_path):
    """Load only the ae_model encoder from a PPO checkpoint dict."""
    import argparse, types
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_args = ckpt['args']  # dict

    env = make_env(ENV_NAME)
    reset = env.reset()
    sample_obs = reset[0] if isinstance(reset, tuple) else reset
    sample_obs = preprocess_obs([sample_obs])
    env.close()

    # Build a minimal namespace that construct_ae_model needs
    ns = argparse.Namespace(
        ae_model_type    = saved_args['ae_model_type'],
        ae_model_version = saved_args['ae_model_version'],
        ae_model_hash    = saved_args['ae_model_hash'],
        embedding_dim    = saved_args['embedding_dim'],
        codebook_size    = saved_args['codebook_size'],
        filter_size      = saved_args['filter_size'],
        commitment_cost  = saved_args.get('commitment_cost', 0.25),
        ema_decay        = saved_args.get('ema_decay', 0.99),
        latent_dim       = saved_args.get('latent_dim', None),
        stochastic       = saved_args.get('stochastic', 'simple'),
        fta_tiles        = saved_args.get('fta_tiles', 20),
        fta_bound_low    = saved_args.get('fta_bound_low', -2.0),
        fta_bound_high   = saved_args.get('fta_bound_high', 2.0),
        fta_eta          = saved_args.get('fta_eta', 0.2),
        repr_sparsity    = saved_args.get('repr_sparsity', 0),
        sparsity_type    = saved_args.get('sparsity_type', 'random'),
        ctx_channels     = saved_args.get('ctx_channels', 64),
        ctx_cond_type    = saved_args.get('ctx_cond_type', 'concat'),
        code_dropout_rate= saved_args.get('code_dropout_rate', 0.0),
        mae_mask_ratio   = saved_args.get('mae_mask_ratio', 0.0),
        mae_patch_size   = saved_args.get('mae_patch_size', 4),
        mae_loss_coef    = saved_args.get('mae_loss_coef', 1.0),
        ctx_aux_coef     = saved_args.get('ctx_aux_coef', 1.0),
        model_dir        = '..',
        load             = False,   # don't load from hash — we'll load state_dict directly
        e2e_loss         = True,
        use_amp          = False,
        ae_recon_loss    = False,
        wandb            = False,
        comet_ml         = False,
        device           = DEVICE,
        learning_rate    = 3e-4,
        trans_learning_rate = 3e-4,
        ae_grad_clip     = 0,
        log_norms        = False,
        extra_info       = None,
        env_name         = ENV_NAME,
        final_latent_dim = 5184,
        trans_model_type = 'discrete',
        trans_model_version = '1',
        trans_model_hash = '',
        trans_hidden     = 256,
        trans_depth      = 3,
        vq_trans_loss_type   = 'mse',
        vq_trans_1d_conv     = False,
        vq_trans_state_snap  = False,
    )

    ae_model, _ = construct_ae_model(sample_obs.shape[1:], ns,
                                     latent_activation=True, load=False)
    ae_model.load_state_dict(ckpt['ae_model_state_dict'], strict=True)
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad = False
    return ae_model.to(DEVICE)


def collect_data(ae_model, n_frames, n_lat, n_side):
    """Random rollout → (embeddings, sem_labels)."""
    env = make_env(ENV_NAME)
    ae_model.eval()
    embeddings, sem_labels = [], []

    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    obs = torch.from_numpy(obs).float()

    for t in range(n_frames):
        with torch.no_grad():
            flat_emb = ae_model.encode(obs.unsqueeze(0).to(DEVICE),
                                       return_quantized=True)  # (1, n_lat*EMB_DIM)
        embeddings.append(flat_emb.cpu().squeeze(0))

        ug = env.unwrapped
        ge = ug.grid.encode()[:, :, 0].copy()
        ge[ug.agent_pos[0], ug.agent_pos[1]] = OBJECT_TO_IDX['agent']
        ge_rm = ge.T  # row-major
        # Resize if grid != latent size (e.g., DoorKey 8×8 → 9×9)
        if ge_rm.shape != (n_side, n_side):
            from PIL import Image as _PIL
            ge_rm = np.array(
                _PIL.fromarray(ge_rm.astype(np.uint8)).resize(
                    (n_side, n_side), resample=0), dtype=np.int64)
        sem_labels.append(torch.tensor(ge_rm.flatten(), dtype=torch.long))

        act = env.action_space.sample()
        step = env.step(act)
        if len(step) == 5:
            next_obs, _, term, trunc, _ = step
            done = term or trunc
        else:
            next_obs, _, done, _ = step
        if done:
            r = env.reset()
            obs = r[0] if isinstance(r, tuple) else r
        else:
            obs = next_obs
        obs = torch.from_numpy(obs).float()
        if (t + 1) % 10_000 == 0:
            print(f'    {t+1}/{n_frames}')

    env.close()
    return torch.stack(embeddings), torch.stack(sem_labels)


def train_probe(embs, labels, n_lat):
    probe = nn.Linear(EMB_DIM, N_CLASSES).to(DEVICE)
    opt   = optim.Adam(probe.parameters(), lr=1e-3)
    N     = embs.shape[0]
    espatial = embs.view(N, EMB_DIM, n_lat).permute(0, 2, 1)  # (N, n_lat, EMB_DIM)

    for epoch in range(PROBE_EPOCHS):
        idx = torch.randperm(N)
        tot_loss, nb = 0.0, 0
        for i in range(0, N, BATCH):
            b = idx[i:i+BATCH]
            be = espatial[b].to(DEVICE)
            bl = labels[b].to(DEVICE)
            B  = be.shape[0]
            logits = probe(be)
            loss = F.cross_entropy(logits.view(B*n_lat, N_CLASSES),
                                   bl.view(B*n_lat))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); nb += 1
        if (epoch+1) % 5 == 0:
            print(f'    epoch {epoch+1:2d}/{PROBE_EPOCHS}  loss={tot_loss/nb:.4f}')
    return probe


def eval_probe(probe, embs, labels, name, n_lat):
    probe.eval()
    N = embs.shape[0]
    esp = embs.view(N, EMB_DIM, n_lat).permute(0, 2, 1)
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for i in range(0, N, 512):
            logits = probe(esp[i:i+512].to(DEVICE))
            all_pred.append(logits.argmax(-1).cpu())
            all_tgt.append(labels[i:i+512])
    pf = torch.cat(all_pred).view(-1)
    tf = torch.cat(all_tgt).view(-1)
    overall = (pf == tf).float().mean().item()
    print(f'\n  [{name}]  overall accuracy: {overall*100:.1f}%')
    present = sorted(tf.unique().tolist())
    per_class = {}
    for c in present:
        m   = tf == c
        acc = (pf[m] == tf[m]).float().mean().item()
        per_class[c] = acc
        print(f'    {OBJECT_NAMES[c]:10s} (id={c:2d}):  {acc*100:5.1f}%  '
              f'({m.sum().item()} samples)')
    return overall, per_class, pf, tf


def show_sample_frame(ae_model, probe, label='model', n_lat=64, n_side=8):
    """
    Run one env step, show the predicted-vs-truth semantic grid as ASCII.
    Symbols: E=empty  W=wall  L=lava  G=goal  A=agent  .=other
    """
    SYM = {0:'?', 1:'·', 2:'W', 3:'F', 4:'D', 5:'K',
           6:'B', 7:'X', 8:'G', 9:'L', 10:'A'}
    env = make_env(ENV_NAME)
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        flat_emb = ae_model.encode(obs_t, return_quantized=True)  # (1, n_lat*EMB_DIM)
        esp = flat_emb.view(1, EMB_DIM, n_lat).permute(0, 2, 1)  # (1, n_lat, EMB_DIM)
        logits = probe(esp)                                        # (1, n_lat, N_CLASSES)
        pred_flat = logits.argmax(-1).cpu().squeeze(0).numpy()    # (n_lat,)

    ug = env.unwrapped
    ge = ug.grid.encode()[:, :, 0].copy()
    ge[ug.agent_pos[0], ug.agent_pos[1]] = OBJECT_TO_IDX['agent']
    ge_rm = ge.T
    if ge_rm.shape != (n_side, n_side):
        from PIL import Image as _PIL
        ge_rm = np.array(
            _PIL.fromarray(ge_rm.astype(np.uint8)).resize(
                (n_side, n_side), resample=0), dtype=np.int64)
    true_flat = ge_rm.flatten()
    env.close()

    print(f'\n  [{label}] sample {n_side}×{n_side} frame   '
          f'(GT | Pred  — correct=✓ wrong=✗)')
    print('       ' + '  '.join(f'c{j}' for j in range(n_side)))
    for row in range(n_side):
        gt_row   = [SYM.get(true_flat[row*n_side+col], '?')
                    for col in range(n_side)]
        pred_row = [SYM.get(pred_flat[row*n_side+col], '?')
                    for col in range(n_side)]
        marks    = ['✓' if pred_flat[row*n_side+col] == true_flat[row*n_side+col]
                    else '✗' for col in range(n_side)]
        gt_s   = '  '.join(gt_row)
        pred_s = '  '.join(
            f'{p}{m}' for p, m in zip(pred_row, marks))
        print(f'  r{row}:  GT=[{gt_s}]  Pred=[{pred_s}]')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f'\n{"="*60}')
        print(f'  Model: {name}')
        print(f'  Checkpoint: {os.path.basename(ckpt_path)}')
        print(f'{"="*60}')

        ae = load_ae_from_checkpoint(ckpt_path)
        # Auto-detect latent grid size from loaded model
        n_lat = int(ae.n_latent_embeds)
        n_side = int(np.round(np.sqrt(n_lat)))
        assert n_side * n_side == n_lat, f'n_latent_embeds={n_lat} is not a perfect square'
        print(f'  Latent grid: {n_side}×{n_side} = {n_lat} tokens')

        print(f'  Collecting {N_FRAMES} frames...')
        embs, labels = collect_data(ae, N_FRAMES, n_lat, n_side)
        print(f'  embeddings: {tuple(embs.shape)}, labels: {tuple(labels.shape)}')

        # class distribution
        flat_lbl = labels.view(-1)
        dist = {OBJECT_NAMES[c]: (flat_lbl==c).sum().item()
                for c in range(N_CLASSES) if (flat_lbl==c).sum() > 0}
        pct  = {k: f'{v/len(flat_lbl)*100:.1f}%' for k, v in dist.items()}
        print(f'  class dist: {pct}')

        print(f'  Training probe...')
        probe = train_probe(embs, labels, n_lat)

        overall, per_class, pf, tf = eval_probe(probe, embs, labels, name, n_lat)
        show_sample_frame(ae, probe, name, n_lat, n_side)
        results[name] = {'overall': overall, 'per_class': per_class}

    # ── comparison summary ────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('  COMPARISON SUMMARY')
    print(f'{"="*60}')
    names = list(results.keys())
    present = sorted({c for r in results.values()
                      for c in r['per_class']})
    hdr = f'  {"class":10s}   ' + '  '.join(f'{n:>18}' for n in names)
    print(hdr)
    for c in present:
        row = f'  {OBJECT_NAMES[c]:10s}   '
        for n in names:
            acc = results[n]['per_class'].get(c, float('nan'))
            row += f'{acc*100:17.1f}%  '
        print(row)
    print(f'  {"OVERALL":10s}   ' +
          '  '.join(f'{results[n]["overall"]*100:17.1f}%  ' for n in names))


if __name__ == '__main__':
    main()
