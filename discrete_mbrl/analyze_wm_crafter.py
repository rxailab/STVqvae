"""
Crafter-specific Phase A analyzer (cross-env generality test for §6).

Adapts analyze_wm_multistep.py to Crafter:
  - Crafter has 19 semantic classes (vs MiniGrid's 11).
  - Per-cell ground-truth grid comes from CrafterGymnasiumWrapper.get_semantic()
    rather than minigrid's env.grid.encode().
  - 64x64 RGB obs, latent grid 8x8 (filter_size=8).
  - 17 actions vs MiniGrid's 7.

Computes the same per-class probe + per-class free-running WM accuracy + Pearson
r the MiniGrid analyzer does. The headline question for the paper: does the
probe-WM dissociation also appear on Crafter?

Usage:
    python analyze_wm_crafter.py --model_path ckpt.pt --output_json out.json
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs  # noqa: E402

# Crafter sem_n_classes = 19 in the ckpt args; we treat IDs 0..18 as classes.
CRAFTER_N_CLASSES = 19
# Best-effort class names from Crafter's source; if not available we fall back to ids.
try:
    import crafter as _crafter_pkg
    if hasattr(_crafter_pkg, 'objects') and hasattr(_crafter_pkg.objects, 'walkable'):
        # crafter has a worldgen vocab; we'll map by index.
        _CR_NAMES = []
    _CRAFTER_NAMES = None
except ImportError:
    _CRAFTER_NAMES = None
# Fallback hardcoded approximation of crafter's standard class set:
_CRAFTER_NAMES = [
    'invalid', 'water', 'grass', 'stone', 'sand', 'lava',
    'coal', 'iron', 'diamond', 'tree', 'wood', 'plant', 'sapling',
    'table', 'furnace', 'player', 'cow', 'zombie', 'skeleton',
][:CRAFTER_N_CLASSES]


def crafter_semantic_grid(env, n_lat_side):
    """Get per-cell semantic class grid from Crafter wrapper, resized to (n_lat_side, n_lat_side)."""
    from PIL import Image as PIL_Image
    # The env may be wrapped (SeedCompatWrapper, TimeLimit, etc.); walk the
    # wrapper chain to find get_semantic on the underlying CrafterGymnasiumWrapper.
    target = env
    while not hasattr(target, 'get_semantic') and hasattr(target, 'env'):
        target = target.env
    if not hasattr(target, 'get_semantic'):
        return np.zeros(n_lat_side * n_lat_side, dtype=np.int64)
    sem = target.get_semantic()
    if sem is None:
        return np.zeros(n_lat_side * n_lat_side, dtype=np.int64)
    sem = np.asarray(sem, dtype=np.uint8)
    # Crafter's _sem_view typically returns the agent's local view
    # (e.g., 7x9 or 64x64 depending on version). Resize to latent grid size.
    if sem.shape != (n_lat_side, n_lat_side):
        sem = np.array(
            PIL_Image.fromarray(sem).resize((n_lat_side, n_lat_side), PIL_Image.NEAREST))
    sem = np.clip(sem, 0, CRAFTER_N_CLASSES - 1)
    return sem.flatten().astype(np.int64)


def encode_indices(ae_model, obs_t, device):
    with torch.no_grad():
        out = ae_model.encode(obs_t.unsqueeze(0).to(device), return_quantized=False)
    if out.dim() == 3:
        out = out.view(1, -1)
    return out.long()


def encode_embeds(ae_model, obs_t, device, enc_type):
    with torch.no_grad():
        if enc_type == 'vq':
            return ae_model.encode(obs_t.unsqueeze(0).to(device), return_quantized=True)
        return ae_model.encode(obs_t.unsqueeze(0).to(device))


def load_checkpoint_crafter(model_path, device):
    """Like analyze_wm_multistep.load_checkpoint but tolerant of crafter env."""
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    margs = types.SimpleNamespace(**ckpt['args'])

    from model_construction import make_ae

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

    ae_state = ckpt['ae_model_state_dict']
    is_vq = any('quantizer' in k for k in ae_state)
    is_vae = any('mean_conv' in k for k in ae_state)

    if is_vq:
        from shared.models.encoder_models import VQVAEModel
        ae_model = VQVAEModel(
            sample_obs.shape[1:],
            codebook_size=getattr(margs, 'codebook_size', 64),
            embedding_dim=getattr(margs, 'embedding_dim', 64),
            encoder=encoder, decoder=decoder,
            commitment_cost=0.25, ema_decay=0.99,
            dead_code_threshold=getattr(margs, 'dead_code_threshold', 0.0),
        )
        enc_type = 'vq'
    elif is_vae:
        from shared.models.encoder_models import AEModelSpatial
        ae_model = AEModelSpatial(
            sample_obs.shape[1:],
            embedding_dim=getattr(margs, 'embedding_dim', 64),
            encoder=encoder, decoder=decoder,
            stochastic=getattr(margs, 'stochastic', True),
        )
        enc_type = 'vae_spatial'
    else:
        raise RuntimeError(f'cannot detect encoder type from state-dict')

    ae_model.load_state_dict(ckpt['ae_model_state_dict'])
    ae_model = ae_model.to(device).eval()

    trans_model = None
    if 'trans_model_state_dict' in ckpt and getattr(margs, 'use_world_model', False):
        try:
            n_latents = ae_model.n_latent_embeds
            emb_dim = getattr(margs, 'embedding_dim', 64)
            hidden = getattr(margs, 'trans_hidden', 256)
            depth = getattr(margs, 'trans_depth', 3)
            if enc_type == 'vq':
                from shared.models.transition_models import DiscreteTransitionModel
                trans_model = DiscreteTransitionModel(
                    n_latents, getattr(margs, 'codebook_size', 512), emb_dim,
                    env.action_space, hidden_sizes=[hidden]*depth,
                    stochastic=False, stoch_hidden_sizes=[256, 256],
                    discretizer_hidden_sizes=[256], use_soft_embeds=False,
                    return_logits=False)
            else:
                from shared.models.transition_models import ContinuousTransitionModel
                trans_model = ContinuousTransitionModel(
                    n_latents * emb_dim, env.action_space,
                    hidden_sizes=[hidden]*depth)
            trans_model.load_state_dict(ckpt['trans_model_state_dict'])
            trans_model = trans_model.to(device).eval()
        except Exception as e:
            print(f'  warn: could not load transition model: {e}')

    return ae_model, trans_model, env, margs, enc_type


def fit_probe(ae_model, env, n_frames, device, n_lat_side, enc_type):
    from sklearn.linear_model import LogisticRegression
    L = ae_model.n_latent_embeds
    D = ae_model.embedding_dim if hasattr(ae_model, 'embedding_dim') else (
        ae_model.latent_dim // L)

    X_list, y_list = [], []
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    obs_t = torch.from_numpy(obs).float()
    for _ in range(n_frames):
        flat = encode_embeds(ae_model, obs_t, device, enc_type).cpu().view(L, D).numpy()
        sem = crafter_semantic_grid(env, n_lat_side)
        X_list.append(flat)
        y_list.append(sem)
        a = env.action_space.sample()
        sr = env.step(a)
        obs = sr[0] if isinstance(sr, tuple) else sr
        done = sr[2] if len(sr) >= 3 else False
        obs_t = torch.from_numpy(obs).float()
        if done:
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
            obs_t = torch.from_numpy(obs).float()

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    probe = LogisticRegression(max_iter=300, class_weight='balanced', C=1.0, n_jobs=-1)
    probe.fit(X, y)
    pred = probe.predict(X)
    per_class = {}
    for c in np.unique(y):
        m = y == c
        n = int(m.sum())
        if n < 10:
            continue
        per_class[int(c)] = {
            'acc': float((pred[m] == c).mean()),
            'count': n,
            'name': _CRAFTER_NAMES[int(c)] if int(c) < len(_CRAFTER_NAMES) else f'cls_{c}',
        }
    return probe, per_class


def freerun_wm(trans_model, ae_model, codes_0, actions, k_max, device, enc_type):
    """Free-run for k_max steps; returns list of predicted code grids (VQ) or
    embeddings (VAE), one per horizon."""
    if enc_type == 'vq':
        codes = torch.from_numpy(codes_0).unsqueeze(0).long().to(device)
        preds = []
        for k in range(k_max):
            a = torch.tensor([int(actions[k])], device=device, dtype=torch.long)
            with torch.no_grad():
                logits, _, _ = trans_model(codes, a, return_logits=True)
            codes = logits.argmax(dim=1)
            preds.append(codes.squeeze(0).cpu().numpy())
        return preds
    else:
        z = torch.from_numpy(codes_0).unsqueeze(0).to(device)
        preds = []
        for k in range(k_max):
            a = torch.tensor([int(actions[k])], device=device, dtype=torch.long)
            with torch.no_grad():
                states, _, _ = trans_model(z, a)
            z = states.view(1, -1)
            preds.append(z.squeeze(0).cpu().numpy())
        return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--probe_frames', type=int, default=8000)
    parser.add_argument('--n_rollouts', type=int, default=200)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()

    print(f'Loading {args.model_path}')
    ae_model, trans_model, env, margs, enc_type = load_checkpoint_crafter(args.model_path, args.device)
    print(f'  env={margs.env_name}  enc={enc_type}  '
          f'L={ae_model.n_latent_embeds}  D={getattr(ae_model, "embedding_dim", "?")}')
    if trans_model is None:
        raise RuntimeError('No transition model in this ckpt.')

    L = ae_model.n_latent_embeds
    n_lat_side = int(round(L ** 0.5))

    print(f'\n[1/3] Probe ({args.probe_frames} random-action frames)')
    env_probe = make_env(margs.env_name)
    probe, probe_acc = fit_probe(ae_model, env_probe, args.probe_frames,
                                 args.device, n_lat_side, enc_type)

    print(f'\n[2/3] Free-running rollouts ({args.n_rollouts} ep, k_max={max(args.horizons)})')
    k_max = max(args.horizons)
    correct_per_k = {k: [] for k in args.horizons}
    sem0_per_k = {k: [] for k in args.horizons}

    for r in range(args.n_rollouts):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()

        sem_t0 = crafter_semantic_grid(env, n_lat_side)
        if enc_type == 'vq':
            codes_0 = encode_indices(ae_model, obs_t, args.device).squeeze(0).cpu().numpy()
            init = codes_0
        else:
            with torch.no_grad():
                z0 = ae_model.encode(obs_t.unsqueeze(0).to(args.device))
            init = z0.view(-1).cpu().numpy()

        # Step env, collecting actions and ground-truth states
        gt_states = []
        actions = []
        for step in range(k_max):
            a = env.action_space.sample()
            actions.append(a)
            sr = env.step(a)
            obs = sr[0] if isinstance(sr, tuple) else sr
            done = sr[2] if len(sr) >= 3 else False
            obs_t = torch.from_numpy(obs).float()
            if enc_type == 'vq':
                gt_states.append(encode_indices(ae_model, obs_t, args.device).squeeze(0).cpu().numpy())
            else:
                with torch.no_grad():
                    z_t = ae_model.encode(obs_t.unsqueeze(0).to(args.device))
                gt_states.append(z_t.view(-1).cpu().numpy())
            if done:
                break
        reached = len(actions)

        if reached < 1:
            continue
        preds = freerun_wm(trans_model, ae_model, init, actions, reached,
                           args.device, enc_type)

        for k in args.horizons:
            if k > reached:
                continue
            if enc_type == 'vq':
                correct = (preds[k - 1] == gt_states[k - 1]).astype(np.float32)
            else:
                # cosine-argmax-within-frame
                D = ae_model.embedding_dim
                p = preds[k - 1].reshape(L, D)
                t = gt_states[k - 1].reshape(L, D)
                p_n = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-9)
                t_n = t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-9)
                sims = p_n @ t_n.T
                correct = (sims.argmax(axis=-1) == np.arange(L)).astype(np.float32)
            correct_per_k[k].append(correct)
            sem0_per_k[k].append(sem_t0)

        if (r + 1) % 50 == 0:
            print(f'  rollouts {r+1}/{args.n_rollouts}')

    # Per-class WM accuracy bucketed by class at t=0
    print('\n[3/3] Scoring')
    wm_per_horizon = {}
    for k in args.horizons:
        if not correct_per_k[k]:
            continue
        c_all = np.concatenate(correct_per_k[k])
        s_all = np.concatenate(sem0_per_k[k])
        per_class = {}
        for cls in range(CRAFTER_N_CLASSES):
            m = s_all == cls
            n = int(m.sum())
            if n < 10:
                continue
            per_class[int(cls)] = {
                'acc': float(c_all[m].mean()),
                'count': n,
                'name': _CRAFTER_NAMES[cls] if cls < len(_CRAFTER_NAMES) else f'cls_{cls}',
            }
        wm_per_horizon[k] = per_class

    # Pearson r per horizon
    pearson = {}
    for k in args.horizons:
        if k not in wm_per_horizon:
            continue
        common = sorted(set(probe_acc.keys()) & set(wm_per_horizon[k].keys()))
        if len(common) < 3:
            continue
        xs = np.array([probe_acc[c]['acc'] for c in common])
        ys = np.array([wm_per_horizon[k][c]['acc'] for c in common])
        if xs.std() < 1e-9 or ys.std() < 1e-9:
            continue
        pearson[k] = float(np.corrcoef(xs, ys)[0, 1])

    # Console
    print(f'\n=== Crafter probe vs WM accuracy ===')
    print(f'  {"class":<14}{"probe":>8}  ' + '  '.join(f'WM{k}'.rjust(7) for k in args.horizons) + '  count')
    for cls in sorted(probe_acc.keys()):
        nm = probe_acc[cls]['name']
        p = probe_acc[cls]['acc']
        cells = []
        for k in args.horizons:
            v = wm_per_horizon.get(k, {}).get(cls, {}).get('acc')
            cells.append(f'{v*100:>6.1f}%' if v is not None else '   --')
        print(f'  {nm:<14}{p*100:>6.1f}%  ' + '  '.join(cells) + f'  {probe_acc[cls]["count"]:>6}')
    print('\nPearson r (probe vs WM-k-step):')
    for k in args.horizons:
        if k in pearson:
            print(f'  k={k:>2}: {pearson[k]:+.3f}')

    result = {
        'model_path': args.model_path,
        'env_name': margs.env_name,
        'encoder_type': enc_type,
        'probe_acc_per_class': {str(c): v for c, v in probe_acc.items()},
        'wm_acc_per_class_per_horizon': {
            str(k): {str(c): v for c, v in d.items()} for k, d in wm_per_horizon.items()
        },
        'pearson_r_per_horizon': {str(k): v for k, v in pearson.items()},
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved {args.output_json}')


if __name__ == '__main__':
    main()
