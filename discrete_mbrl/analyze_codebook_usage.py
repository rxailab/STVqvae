"""
Codebook usage analysis — produces code-to-class assignment matrix.

For each encoder variant, shows:
  - How many codes are allocated to each object class
  - What fraction of codes are dead
  - Entropy of code-class assignment matrix
  - Dominant-class purity per code

Usage:
    python analyze_codebook_usage.py \
        --model_path model_free/models/MiniGrid-DoorKey-8x8-v0/mf_e2e_semantic_doorkey_v6enc_goal_best_model.pt \
        --n_frames 10000 --device cuda

    # Or for Crafter:
    python analyze_codebook_usage.py \
        --model_path model_free/models/crafter/mf_e2e_semantic_crafter_v6enc_fix_best_model.pt \
        --n_frames 10000 --device cuda
"""

import argparse, os, sys, types, json
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env, preprocess_obs, OBJECT_TO_IDX
from model_construction import construct_ae_model

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

CRAFTER_CLASSES = {0:'invalid',1:'water',2:'grass',3:'stone',4:'path',5:'sand',
                   6:'tree',7:'lava',8:'coal',9:'iron',10:'diamond',11:'table',
                   12:'furnace',13:'plant',14:'fence',15:'player',16:'cow',
                   17:'zombie',18:'skeleton'}


def get_class_name(class_id, is_crafter):
    if is_crafter:
        return CRAFTER_CLASSES.get(class_id, f'cls{class_id}')
    return IDX_TO_OBJECT.get(class_id, f'cls{class_id}')


def collect_code_class_pairs(env, ae_model, n_frames, device, n_lat_side, is_crafter):
    """Collect (code_index, semantic_class) pairs from random rollout."""
    ae_model.eval()
    pairs = []

    obs, _ = env.reset()
    obs_t = preprocess_obs([obs]).to(device)

    with torch.no_grad():
        for step in range(n_frames):
            enc_out = ae_model.encode(obs_t, return_quantized=False)
            if enc_out.ndim == 3:
                idx = enc_out.argmax(-1).squeeze(0).cpu().numpy()
            elif enc_out.ndim == 2:
                idx = enc_out.squeeze(0).cpu().numpy()
            else:
                idx = enc_out.view(-1).cpu().numpy()

            # Get semantic map
            inner = env
            while hasattr(inner, 'env'):
                inner = inner.env

            if is_crafter:
                sem_map = None
                if hasattr(inner, 'get_semantic'):
                    sem_map = inner.get_semantic()
                elif hasattr(inner, '_sem_view'):
                    sem_map = inner._sem_view()
                if sem_map is None:
                    obs, _, t, tr, _ = env.step(env.action_space.sample())
                    if t or tr: obs, _ = env.reset()
                    obs_t = preprocess_obs([obs]).to(device)
                    continue
                ge = sem_map
            else:
                ge = inner.grid.encode()[:, :, 0].copy()
                ge[inner.agent_pos[0], inner.agent_pos[1]] = OBJECT_TO_IDX['agent']
                ge = ge.T  # col-major → row-major

            # Resize to match latent grid
            from PIL import Image as _PIL
            if ge.shape != (n_lat_side, n_lat_side):
                ge = np.array(
                    _PIL.fromarray(ge.astype(np.uint8)).resize(
                        (n_lat_side, n_lat_side), resample=0),
                    dtype=np.int64)

            flat_sem = ge.flatten()
            for code, cls in zip(idx.flatten(), flat_sem):
                pairs.append((int(code), int(cls)))

            action = env.action_space.sample()
            obs, _, t, tr, _ = env.step(action)
            if t or tr: obs, _ = env.reset()
            obs_t = preprocess_obs([obs]).to(device)

            if (step + 1) % 5000 == 0:
                print(f'  {step+1}/{n_frames} frames collected')

    return pairs


def analyze(pairs, codebook_size, is_crafter):
    """Compute code-class assignment matrix and statistics."""
    # Build joint count matrix: (codebook_size, n_classes)
    all_classes = sorted(set(cls for _, cls in pairs))
    n_classes = max(all_classes) + 1
    joint = np.zeros((codebook_size, n_classes), dtype=np.int64)
    for code, cls in pairs:
        joint[code, cls] += 1

    # Marginals
    code_counts = joint.sum(axis=1)      # (K,) how often each code is used
    class_counts = joint.sum(axis=0)     # (C,) how often each class appears
    total = joint.sum()

    # Dead codes (never used)
    dead_mask = code_counts == 0
    n_dead = dead_mask.sum()
    n_active = codebook_size - n_dead

    # Per-code: dominant class and purity
    dominant_class = np.argmax(joint, axis=1)
    purity = np.zeros(codebook_size)
    for k in range(codebook_size):
        if code_counts[k] > 0:
            purity[k] = joint[k, dominant_class[k]] / code_counts[k]

    # Per-class: how many codes primarily serve this class
    class_code_count = Counter()
    for k in range(codebook_size):
        if code_counts[k] > 0:
            class_code_count[dominant_class[k]] += 1

    # Entropy of P(class | code) averaged over codes
    # H(class | code) = sum_k P(k) * H(class | code=k)
    h_class_given_code = 0.0
    for k in range(codebook_size):
        if code_counts[k] == 0:
            continue
        p_k = code_counts[k] / total
        p_c_given_k = joint[k] / code_counts[k]
        p_c_given_k = p_c_given_k[p_c_given_k > 0]
        h_k = -np.sum(p_c_given_k * np.log2(p_c_given_k))
        h_class_given_code += p_k * h_k

    # Entropy of P(code | class) averaged over classes
    h_code_given_class = 0.0
    for c in all_classes:
        if class_counts[c] == 0:
            continue
        p_c = class_counts[c] / total
        p_k_given_c = joint[:, c] / class_counts[c]
        p_k_given_c = p_k_given_c[p_k_given_c > 0]
        h_c = -np.sum(p_k_given_c * np.log2(p_k_given_c))
        h_code_given_class += p_c * h_c

    # Marginal entropy H(code)
    p_code = code_counts / total
    p_code = p_code[p_code > 0]
    h_code = -np.sum(p_code * np.log2(p_code))

    return {
        'joint': joint,
        'code_counts': code_counts,
        'class_counts': class_counts,
        'n_dead': int(n_dead),
        'n_active': int(n_active),
        'codebook_size': codebook_size,
        'dead_fraction': n_dead / codebook_size,
        'dominant_class': dominant_class,
        'purity': purity,
        'avg_purity': float(purity[~dead_mask].mean()) if n_active > 0 else 0,
        'class_code_count': dict(class_code_count),
        'h_class_given_code': float(h_class_given_code),
        'h_code_given_class': float(h_code_given_class),
        'h_code': float(h_code),
        'all_classes': all_classes,
    }


def print_results(stats, is_crafter):
    K = stats['codebook_size']
    print(f'\n{"="*60}')
    print(f'CODEBOOK USAGE ANALYSIS')
    print(f'{"="*60}')
    print(f'  Codebook size        : {K}')
    print(f'  Active codes         : {stats["n_active"]}/{K} ({stats["n_active"]/K*100:.1f}%)')
    print(f'  Dead codes           : {stats["n_dead"]}/{K} ({stats["dead_fraction"]*100:.1f}%)')
    print(f'  Avg code purity      : {stats["avg_purity"]*100:.1f}%')
    print(f'  H(class|code)        : {stats["h_class_given_code"]:.3f} bits  (lower = codes are class-pure)')
    print(f'  H(code|class)        : {stats["h_code_given_class"]:.3f} bits  (lower = each class uses few codes)')
    print(f'  H(code) [marginal]   : {stats["h_code"]:.3f} bits  (max={np.log2(K):.1f})')

    print(f'\n  Code allocation per class:')
    print(f'    {"Class":12s}  {"# Codes":>8s}  {"Data %":>7s}')
    print(f'    {"-"*32}')
    for c in stats['all_classes']:
        n_codes = stats['class_code_count'].get(c, 0)
        data_pct = stats['class_counts'][c] / stats['class_counts'].sum() * 100
        name = get_class_name(c, is_crafter)
        print(f'    {name:12s}  {n_codes:8d}  {data_pct:6.1f}%')

    # Top-10 most used codes
    print(f'\n  Top 10 most-used codes:')
    top_codes = np.argsort(-stats['code_counts'])[:10]
    for k in top_codes:
        if stats['code_counts'][k] == 0:
            break
        dom_cls = stats['dominant_class'][k]
        name = get_class_name(dom_cls, is_crafter)
        print(f'    Code {k:4d}: {stats["code_counts"][k]:7d} uses, '
              f'dominant={name} ({stats["purity"][k]*100:.0f}% pure)')

    print(f'{"="*60}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_frames', type=int, default=10000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None, help='Save stats to JSON')
    args = parser.parse_args()

    print(f'Loading: {args.model_path}')
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    margs = types.SimpleNamespace(**ckpt['args'])
    margs.load = False; margs.e2e_loss = True; margs.use_amp = False
    margs.ae_recon_loss = getattr(margs, 'ae_recon_loss', False)
    margs.dead_code_threshold = getattr(margs, 'dead_code_threshold', 0.0)

    is_crafter = 'crafter' in margs.env_name.lower()
    print(f'Env: {margs.env_name} (crafter={is_crafter})')
    print(f'Codebook: {margs.codebook_size}, Embed: {margs.embedding_dim}')

    env = make_env(margs.env_name)
    obs, _ = env.reset()
    sample_obs = preprocess_obs([obs])

    ae_model, _ = construct_ae_model(sample_obs.shape[1:], margs,
                                     latent_activation=True, load=False)
    ae_model.load_state_dict(ckpt['ae_model_state_dict'])
    ae_model = ae_model.to(args.device).eval()
    for p in ae_model.parameters(): p.requires_grad = False

    n_latents = ae_model.n_latent_embeds
    n_lat_side = int(round(n_latents ** 0.5))

    # EMA utilization
    ema_usage = ae_model.quantizer._ema_cluster_size.detach().cpu().numpy()
    n_ema_active = (ema_usage > 0.5).sum()
    print(f'EMA utilization: {n_ema_active}/{margs.codebook_size}')

    print(f'\nCollecting {args.n_frames} frames...')
    pairs = collect_code_class_pairs(env, ae_model, args.n_frames, args.device,
                                     n_lat_side, is_crafter)
    print(f'Collected {len(pairs):,} (code, class) pairs')

    stats = analyze(pairs, margs.codebook_size, is_crafter)
    print_results(stats, is_crafter)

    if args.output_json:
        # Save serializable version
        out = {k: (int(v) if isinstance(v, (np.integer,)) else v)
               for k, v in stats.items()
               if k not in ('joint', 'code_counts', 'class_counts', 'dominant_class', 'purity')}
        # Convert numpy int keys in class_code_count to str
        out['class_code_count'] = {str(k): v for k, v in stats['class_code_count'].items()}
        out['all_classes'] = [int(c) for c in stats['all_classes']]
        out['model_path'] = args.model_path
        out['env_name'] = margs.env_name
        out['codebook_size'] = margs.codebook_size
        out['embedding_dim'] = margs.embedding_dim
        out['ae_model_version'] = getattr(margs, 'ae_model_version', 'unknown')
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSaved to {args.output_json}')


if __name__ == '__main__':
    main()
