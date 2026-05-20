"""
Phase B (E7): Decoder-cycle probe.

For each predicted ẑₖ from a free-running WM rollout, run through the trained
decoder to a rendered RGB frame, then re-encode that frame through the same
encoder, then apply the SAME shared probe β used in Phase A's E1.

The metric tests whether the WM's predicted embedding -- after a decode-encode
cycle that filters it through the model's own input distribution -- still
preserves class information that the probe can read.

Three bookkeeping accuracies are reported per horizon, per class:
  E1            -- β applied directly to ẑₖ                (Phase A baseline)
  E7_pre        -- β applied to encoder(decoder(ẑₖ))        (decoder-cycle metric)
  E7_self       -- β applied to encoder(decoder(z_star_k))  (cycle's own ceiling)

E7_pre = E1 means the decoder-encode cycle is a no-op on WM outputs (no info
gained or lost). E7_pre > E1 means the decoder + encoder together "snap" the
WM's embedding back to a probe-readable point. E7_pre << E1 means the rendering
loses semantic content the embedding had.

Usage:
    python analyze_wm_phaseB.py --model_path ckpt.pt --output_json out.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env  # noqa: E402

from analyze_wm_multistep import (  # noqa: E402
    load_checkpoint, semantic_grid, encode_codes, encode_latent,
    rollout_discrete, rollout_continuous,
    fit_shared_probe,
    OBJECT_NAMES, N_CLASSES,
)


# ── Layout helpers ─────────────────────────────────────────────────────────
# The encoder's flat output (B, D*H*W) is channel-major: flat[c*H*W + h*W + w] =
# z[c, h, w]. The continuous WM (VAE path) preserves that layout. The discrete
# WM produces (B, L) code indices; codebook lookup gives (B, L, D) where the
# `L`-axis is position-major. To feed either path into the decoder we must
# convert to a (B, D, H, W) tensor with the channel-major flat that matches
# the encoder's output layout.


def codes_to_spatial(codes_BL, codebook_KD, encoder_out_shape, device):
    """codes_BL: (B, L) long. codebook: (K, D). Returns (B, D, H, W) on device."""
    D, H, W = encoder_out_shape
    B, L = codes_BL.shape
    assert L == H * W, f'code grid size {L} != H*W ({H}*{W})'
    z_BLD = codebook_KD[codes_BL]               # (B, L, D)
    z_BDL = z_BLD.permute(0, 2, 1)               # (B, D, L)
    return z_BDL.reshape(B, D, H, W).to(device)


def vae_flat_to_spatial(z_flat_BD, encoder_out_shape, device):
    """z_flat: (B, D*H*W) channel-major. Returns (B, D, H, W) on device."""
    D, H, W = encoder_out_shape
    return z_flat_BD.reshape(z_flat_BD.shape[0], D, H, W).to(device)


# ── Decoder-cycle ──────────────────────────────────────────────────────────

def decode(ae_model, z_spatial_BDHW):
    """Run the decoder, handling both VQ and VAE wrappers."""
    if hasattr(ae_model, 'decoder') and ae_model.decoder is not None:
        return ae_model.decoder(z_spatial_BDHW)
    if hasattr(ae_model, 'decode'):
        return ae_model.decode(z_spatial_BDHW)
    raise RuntimeError('ae_model has neither .decoder nor .decode')


def encode_flat(ae_model, img_B3HW, trans_kind):
    """Re-encode a rendered frame to the flat continuous embedding the probe expects."""
    if trans_kind == 'discrete':
        try:
            return ae_model.encode(img_B3HW, return_quantized=True)
        except TypeError:
            return ae_model.encode(img_B3HW)
    return ae_model.encode(img_B3HW)


def cycle_predict(ae_model, probe, z_spatial_BDHW, L, embedding_dim, trans_kind, device):
    """Decode -> re-encode -> probe. Returns (B, L) predicted class array."""
    with torch.no_grad():
        img = decode(ae_model, z_spatial_BDHW)
        re_flat = encode_flat(ae_model, img, trans_kind)            # (B, D*L)
    re_flat = re_flat.cpu().numpy()
    B = re_flat.shape[0]
    # Use the same .view(L, D) convention the probe was fit with (POS-major view
    # of a CHANNEL-major flat). Self-consistent because we apply the same view
    # in both fit and predict.
    re_LD = re_flat.reshape(B * L, embedding_dim)
    pred = probe.predict(re_LD).reshape(B, L)
    return pred


def per_class_recall(pred_RL, true_RL):
    correct = (pred_RL == true_RL).astype(np.float32).reshape(-1)
    classes = true_RL.reshape(-1)
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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_rollouts', type=int, default=300)
    parser.add_argument('--probe_frames', type=int, default=10000)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()

    print(f'Loading {args.model_path}')
    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_model is None:
        raise RuntimeError('No transition model in checkpoint -- cannot run Phase B.')

    L = ae_model.n_latent_embeds
    n_lat_side = int(round(L ** 0.5))
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L
    encoder_out_shape = getattr(ae_model, 'encoder_out_shape', None)
    if encoder_out_shape is None:
        encoder_out_shape = (embedding_dim, n_lat_side, n_lat_side)
    if len(encoder_out_shape) != 3:
        raise RuntimeError(f'unexpected encoder_out_shape: {encoder_out_shape}')

    # Probe (E1's β)
    print(f'Fitting shared probe ({args.probe_frames} frames)')
    env_probe = make_env(margs.env_name)
    probe, _, _, _, probe_acc = fit_shared_probe(
        ae_model, env_probe, args.probe_frames, args.device, n_lat_side, trans_kind)

    # Codebook tensor (VQ only) for code -> embedding lookup
    codebook = None
    if trans_kind == 'discrete':
        codebook = ae_model.quantizer._embedding.weight.detach().to(args.device)

    # Collect rollouts
    print(f'Collecting {args.n_rollouts} rollouts up to k_max={max(args.horizons)}')
    k_max = max(args.horizons)

    # Per-horizon: lists of (predicted z spatial), (z_star spatial), (sem at t=0)
    pred_spatial = {k: [] for k in args.horizons}
    star_spatial = {k: [] for k in args.horizons}
    pred_emb_LD  = {k: [] for k in args.horizons}    # for E1 ground-truth comparison
    sem0         = {k: [] for k in args.horizons}

    for r in range(args.n_rollouts):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()

        sem_t0 = semantic_grid(env, n_lat_side).numpy()                 # (L,)

        # initial encoding
        if trans_kind == 'discrete':
            codes_0 = encode_codes(ae_model, obs_t, args.device)         # (L,)
        else:
            with torch.no_grad():
                z0_flat = ae_model.encode(obs_t.unsqueeze(0).to(args.device))
            z0_flat = z0_flat.view(-1)                                   # (D*L,)

        # ground-truth latent at each horizon (for E7_self comparison)
        gt_lat_per_step = {0: None}                                      # filled lazily

        actions = []
        # Step the env first, recording sem at each step (for the gt_lat encoding)
        gt_obs_at = {0: obs_t}
        reached = 0
        for step in range(k_max):
            a = env.action_space.sample()
            actions.append(a)
            sr = env.step(a)
            done = sr[2] if len(sr) >= 3 else False
            obs = sr[0] if isinstance(sr, tuple) else sr
            obs_t_next = torch.from_numpy(obs).float()
            gt_obs_at[step + 1] = obs_t_next
            reached = step + 1
            if done:
                break

        if reached < 1:
            continue

        # WM rollout
        if trans_kind == 'discrete':
            preds = rollout_discrete(trans_model, codes_0, actions[:reached], args.device)
            # preds: list of (L,) long tensors
            pred_spatial_per_k = []
            for p in preds:
                p_BL = p.unsqueeze(0).to(args.device)                     # (1, L) long
                pred_spatial_per_k.append(
                    codes_to_spatial(p_BL, codebook, encoder_out_shape, args.device)
                )
            # ground-truth spatial via re-encoding
            star_spatial_per_k = []
            for k_step in range(1, reached + 1):
                with torch.no_grad():
                    obs_in = gt_obs_at[k_step].unsqueeze(0).to(args.device)
                    flat = ae_model.encode(obs_in, return_quantized=True)  # (1, D*L)
                star_spatial_per_k.append(
                    vae_flat_to_spatial(flat, encoder_out_shape, args.device)
                )
            # also save the (L, D) view used by the per-token probe for E1 comparison
            pred_emb_LD_per_k = []
            for p in preds:
                # POS-major (L, D): codebook[p] is (L, D) directly
                pred_emb_LD_per_k.append(codebook[p.to(args.device)].cpu().numpy())
        else:
            preds = rollout_continuous(trans_model, ae_model, z0_flat, actions[:reached], args.device)
            # preds: list of (D*L,) flat tensors
            pred_spatial_per_k = []
            pred_emb_LD_per_k = []
            for p in preds:
                p_flat = p.unsqueeze(0).to(args.device)                   # (1, D*L)
                pred_spatial_per_k.append(
                    vae_flat_to_spatial(p_flat, encoder_out_shape, args.device)
                )
                # Self-consistent (L, D) view for E1 baseline (matches probe-fit convention)
                pred_emb_LD_per_k.append(p.cpu().numpy().reshape(L, embedding_dim))
            star_spatial_per_k = []
            for k_step in range(1, reached + 1):
                with torch.no_grad():
                    obs_in = gt_obs_at[k_step].unsqueeze(0).to(args.device)
                    flat = ae_model.encode(obs_in)
                star_spatial_per_k.append(
                    vae_flat_to_spatial(flat, encoder_out_shape, args.device)
                )

        for k in args.horizons:
            if k > reached:
                continue
            pred_spatial[k].append(pred_spatial_per_k[k - 1].cpu().numpy())
            star_spatial[k].append(star_spatial_per_k[k - 1].cpu().numpy())
            pred_emb_LD[k].append(pred_emb_LD_per_k[k - 1])
            sem0[k].append(sem_t0)

        if (r + 1) % 50 == 0:
            print(f'  rollouts: {r + 1}/{args.n_rollouts}')

    # Score per horizon
    e7_pre   = {}     # decoder cycle on WM prediction
    e7_self  = {}     # decoder cycle on z*_k (the cycle's own ceiling)
    e1_recall = {}    # baseline: probe on raw ẑₖ -- recomputed here with same probe to align Ns

    print('\n=== E7 decoder-cycle probe ===')
    for k in args.horizons:
        if not pred_spatial[k]:
            continue
        ps = np.concatenate(pred_spatial[k], axis=0)              # (R, D, H, W)
        ss = np.concatenate(star_spatial[k], axis=0)
        s0 = np.stack(sem0[k])                                    # (R, L)
        # WM-cycle probe
        ps_t = torch.from_numpy(ps).to(args.device)
        ss_t = torch.from_numpy(ss).to(args.device)
        BATCH = 32
        cycle_pred_wm = []
        cycle_pred_st = []
        for start in range(0, ps.shape[0], BATCH):
            end = min(start + BATCH, ps.shape[0])
            cycle_pred_wm.append(cycle_predict(
                ae_model, probe, ps_t[start:end], L, embedding_dim, trans_kind, args.device))
            cycle_pred_st.append(cycle_predict(
                ae_model, probe, ss_t[start:end], L, embedding_dim, trans_kind, args.device))
        cycle_pred_wm = np.concatenate(cycle_pred_wm, axis=0)
        cycle_pred_st = np.concatenate(cycle_pred_st, axis=0)
        e7_pre[k] = per_class_recall(cycle_pred_wm, s0)
        e7_self[k] = per_class_recall(cycle_pred_st, s0)

        # E1 baseline on the same R subset (for an apples-to-apples comparison
        # with the cycle metric).
        emb_LD = np.stack(pred_emb_LD[k])                          # (R, L, D)
        flat = emb_LD.reshape(-1, embedding_dim)
        e1_pred = probe.predict(flat).reshape(emb_LD.shape[0], L)
        e1_recall[k] = per_class_recall(e1_pred, s0)

        # Console
        names = ['wall', 'door', 'key', 'goal', 'agent']
        line_e1   = ', '.join(f'{n}={(e1_recall[k].get(_n2id(n), {}) or {}).get("acc", float("nan")):.2f}' for n in names)
        line_pre  = ', '.join(f'{n}={(e7_pre[k].get(_n2id(n), {}) or {}).get("acc", float("nan")):.2f}' for n in names)
        line_self = ', '.join(f'{n}={(e7_self[k].get(_n2id(n), {}) or {}).get("acc", float("nan")):.2f}' for n in names)
        print(f' k={k}  E1   : {line_e1}')
        print(f'        E7_pre: {line_pre}')
        print(f'        E7_self: {line_self}')

    # Save
    result = {
        'model_path': args.model_path,
        'trans_model_type': trans_kind,
        'env_name': margs.env_name,
        'horizons': args.horizons,
        'n_rollouts': args.n_rollouts,
        'probe_acc_per_class': {str(c): v for c, v in probe_acc.items()},
        'E1_baseline_per_horizon': {str(k): {str(c): v for c, v in d.items()} for k, d in e1_recall.items()},
        'E7_pre_per_horizon':      {str(k): {str(c): v for c, v in d.items()} for k, d in e7_pre.items()},
        'E7_self_per_horizon':     {str(k): {str(c): v for c, v in d.items()} for k, d in e7_self.items()},
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved {args.output_json}')


def _n2id(name):
    """Map class name to its index for table lookup."""
    try:
        return OBJECT_NAMES.index(name)
    except ValueError:
        return -1


if __name__ == '__main__':
    main()
