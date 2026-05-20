"""Buffer-coverage analysis: does the §4 WM training buffer (random-action
rollouts) actually contain transitions where door/key/goal state changes?

The hypothesis after Phase I: within-regime per-class WM failure is driven
by *zero or near-zero* interaction transitions in the random-action buffer.
A 5M-step random policy almost never picks up a key, even if it walks past
one — pickup requires being adjacent + facing + pressing pickup at the same
time, which happens with very small probability under uniform actions.

This script collects N transitions under each of {random, policy} action
schedules and counts class-changing transitions per class. If door/key/goal
class-change counts are ~0 under random but nonzero under policy, the
per-class within-regime failure is explained by data starvation, not
encoder/WM architecture.

For each model_path:
  - Random rollout: N steps, count class transitions at each position.
  - Policy rollout: N steps using the ckpt's saved policy, same counts.
  - Output: counts table per class, plus first-occurrence times.
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from analyze_wm_multistep import load_checkpoint
from analyze_wm_phaseB_e8 import build_mlp, encode_flat
from analyze_wm_action_cond_v2 import _semantic_grid
from env_helpers import OBJECT_TO_IDX

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
N_CLASSES = 11
OBJECT_NAMES = [IDX_TO_OBJECT.get(i, f'class_{i}') for i in range(N_CLASSES)]
CLASSES_OF_INTEREST = [1, 2, 4, 5, 8, 10]  # empty, wall, door, key, goal, agent


def rollout_and_count(env, ae_model, policy, n_steps, device, n_lat_side, trans_kind):
    """Collect n_steps transitions, count class-changes at each position.

    Returns:
        change_counts: dict {class_id: int}  # count of positions where this
                       class was at t but different class at t+1
        first_change:  dict {class_id: int}  # first step where each class
                       saw a change (-1 if never)
        total_positions: dict {class_id: int}  # total visits to each class
                       across all (step, position) pairs (denominator).
    """
    change_counts = {c: 0 for c in CLASSES_OF_INTEREST}
    first_change = {c: -1 for c in CLASSES_OF_INTEREST}
    total_positions = {c: 0 for c in CLASSES_OF_INTEREST}

    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    obs_t = torch.from_numpy(obs).float()
    sem_prev = _semantic_grid(env, n_lat_side)
    n_actions = env.action_space.n

    for t in range(n_steps):
        # Choose action.
        if policy is not None:
            z = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0)
            with torch.no_grad():
                logits = policy(z.unsqueeze(0))
                probs = F.softmax(logits.squeeze(0), dim=-1)
                a = int(torch.multinomial(probs, 1).item())
        else:
            a = int(np.random.randint(n_actions))

        sr = env.step(a)
        obs_next = sr[0] if isinstance(sr, tuple) else sr
        done = sr[2] if len(sr) >= 3 else False
        obs_t = torch.from_numpy(obs_next).float()
        sem_curr = _semantic_grid(env, n_lat_side)

        # Count class-changes at each position.
        for c in CLASSES_OF_INTEREST:
            prev_mask = (sem_prev == c)
            total_positions[c] += int(prev_mask.sum())
            # positions where prev==c and curr != c
            n_changes = int(((sem_prev == c) & (sem_curr != c)).sum())
            change_counts[c] += n_changes
            if n_changes > 0 and first_change[c] == -1:
                first_change[c] = t

        sem_prev = sem_curr
        if done:
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
            obs_t = torch.from_numpy(obs).float()
            sem_prev = _semantic_grid(env, n_lat_side)

    return change_counts, first_change, total_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_steps', type=int, default=20000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Loading {args.model_path}')
    ae_model, _trans, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)

    L = ae_model.n_latent_embeds
    n_lat_side = int(round(np.sqrt(L)))
    if n_lat_side * n_lat_side != L:
        raise RuntimeError(f'n_latent_embeds={L} not square.')
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    input_dim = L * embedding_dim
    n_actions = env.action_space.n

    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    policy = None
    if 'policy_state_dict' in src and isinstance(src.get('args'), dict) and 'policy_hidden' in src['args']:
        try:
            policy = build_mlp(input_dim, src['args'].get('policy_hidden', [256, 256]),
                               n_actions, 'relu').to(args.device).eval()
            policy.load_state_dict(src['policy_state_dict'])
        except Exception as e:
            print(f'  (policy load failed: {e})')
            policy = None

    print(f'\n[1/2] Random rollout: {args.n_steps} steps')
    rand_changes, rand_first, rand_total = rollout_and_count(
        env, ae_model, None, args.n_steps, args.device, n_lat_side, trans_kind)

    result = {
        'model_path': args.model_path,
        'env_name': margs.env_name,
        'n_steps': args.n_steps,
        'random': {'changes': rand_changes, 'first_change_step': rand_first,
                   'total_positions': rand_total},
        'policy': None,
    }

    print(f'\n  Random changes: ' +
          ' '.join(f'{OBJECT_NAMES[c]}={rand_changes[c]}/{rand_total[c]}'
                   for c in CLASSES_OF_INTEREST))

    if policy is not None:
        print(f'\n[2/2] Policy rollout: {args.n_steps} steps')
        pol_changes, pol_first, pol_total = rollout_and_count(
            env, ae_model, policy, args.n_steps, args.device, n_lat_side, trans_kind)
        result['policy'] = {'changes': pol_changes, 'first_change_step': pol_first,
                            'total_positions': pol_total}
        print(f'\n  Policy changes: ' +
              ' '.join(f'{OBJECT_NAMES[c]}={pol_changes[c]}/{pol_total[c]}'
                       for c in CLASSES_OF_INTEREST))

        # The key ratio: how much more often does each class transition under
        # policy vs random?
        print('\n  Policy/random change-rate ratio per class:')
        for c in CLASSES_OF_INTEREST:
            rand_rate = rand_changes[c] / max(rand_total[c], 1)
            pol_rate = pol_changes[c] / max(pol_total[c], 1)
            ratio = (pol_rate / rand_rate) if rand_rate > 0 else float('inf')
            print(f'    {OBJECT_NAMES[c]:>8s}: random={rand_rate:.5f}  policy={pol_rate:.5f}  ratio={ratio:.1f}x')

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        # JSON-serialise int keys to strings
        def jsonify(d):
            return {str(k): v for k, v in d.items()}
        out = dict(result)
        out['random'] = {k: jsonify(v) for k, v in result['random'].items()}
        if result['policy'] is not None:
            out['policy'] = {k: jsonify(v) for k, v in result['policy'].items()}
        with open(args.output_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSaved {args.output_json}')


if __name__ == '__main__':
    main()
