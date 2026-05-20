"""
Phase N: action-conditioning probe.

Phase M observed that the WM is action-agnostic (q_action_var ~ 0.0007, top1
agreement with policy at chance level for 6 actions). This analyzer tests the
mechanism directly by measuring how much T(z, a) varies with a.

For each visited state z, we compute T(z, a) for every discrete action and
report three geometry metrics:

  state_change_norm   = ||T(z, a) - z||           (mean over a, z)
                        How much the WM moves the state under any action.
  action_diff_norm    = ||T(z, a) - T(z, a')||    (mean over a≠a', z)
                        How much different actions differ in their predictions.
  action_dep_ratio    = action_diff_norm / state_change_norm
                        ≈0 ⇒ actions ignored; ≥1 ⇒ actions strongly
                        differentiate predictions.

  reward_action_var   = Var_a(r̂(z, a))             (mean over z)
  gamma_action_var    = Var_a(γ̂(z, a))             (mean over z)
                        Whether reward and discount heads condition on action.

  effective_action_rank = rank of [T(z,a₁)-z, ..., T(z,a_K)-z] (mean over z)
                        How many distinct directions the WM can predict.
                        =1 means the action just scales a single direction;
                        =n_actions means each action has its own direction.

Usage:
    python analyze_wm_action_cond.py --model_path ckpt.pt --output_json out.json
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from analyze_wm_multistep import load_checkpoint, encode_codes  # noqa: E402
from analyze_wm_phaseB_e8 import build_mlp, encode_flat  # noqa: E402


def collect_states(env, ae_model, policy, n_steps, device, trans_kind, action_sampler='random'):
    """Returns z (continuous flat) and codes (discrete, only for VQ models).
    `policy` is optional — if None, uses random actions (useful for discrete
    WMs whose policy may live in a different latent space)."""
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    obs_t = torch.from_numpy(obs).float()
    z_list, code_list = [], []
    n_actions = env.action_space.n
    for t in range(n_steps):
        z = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
        z_list.append(z.astype(np.float32))
        if trans_kind == 'discrete':
            c = encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
            code_list.append(c.astype(np.int64))
        if policy is not None:
            with torch.no_grad():
                logits = policy(torch.from_numpy(z).unsqueeze(0).to(device))
                probs = F.softmax(logits.squeeze(0), dim=-1)
                a = int(torch.multinomial(probs, 1).item())
        else:
            a = int(np.random.randint(n_actions))
        sr = env.step(a)
        obs_next = sr[0] if isinstance(sr, tuple) else sr
        done = sr[2] if len(sr) >= 3 else False
        obs_t = torch.from_numpy(obs_next).float()
        if done:
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
            obs_t = torch.from_numpy(obs).float()
    return np.stack(z_list), (np.stack(code_list) if code_list else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_states', type=int, default=2000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()

    print(f'Loading {args.model_path}')
    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_model is None:
        raise RuntimeError('No transition model.')

    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    input_dim = L * embedding_dim
    n_actions = env.action_space.n

    # Some discrete VQ ckpts (e.g. Crafter) trained an end-to-end policy in a
    # different latent space; we don't always have a usable PPO policy here.
    # If the ckpt has 'policy_state_dict' load it, otherwise sample states with
    # random actions — Phase N is about T(z, a) geometry, not policy fidelity.
    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    policy = None
    if 'policy_state_dict' in src and isinstance(src.get('args'), dict) and 'policy_hidden' in src['args']:
        try:
            policy = build_mlp(input_dim, src['args'].get('policy_hidden', [256, 256]),
                               n_actions, 'relu').to(args.device).eval()
            policy.load_state_dict(src['policy_state_dict'])
        except Exception as e:
            print(f'  (policy load failed: {e}; falling back to random-action state collection)')
            policy = None

    print(f'\n[1/2] Collecting {args.n_states} states '
          f'({"policy" if policy is not None else "random"} actions; trans_kind={trans_kind})')
    z_arr, code_arr = collect_states(env, ae_model, policy, args.n_states,
                                     args.device, trans_kind)
    print(f'  collected z shape: {z_arr.shape}  n_actions: {n_actions}')
    if code_arr is not None:
        print(f'  collected codes shape: {code_arr.shape}')

    # ------ [2/2] Compute T(z, a) for all states × all actions ------
    print(f'\n[2/2] Computing T(z, a) for all {len(z_arr)} states × {n_actions} actions')
    Tza   = np.zeros((len(z_arr), n_actions, input_dim), dtype=np.float32)
    R     = np.zeros((len(z_arr), n_actions), dtype=np.float32)
    Gamma = np.zeros((len(z_arr), n_actions), dtype=np.float32)

    bs = 256  # batch over states
    if trans_kind == 'discrete':
        # VQ: codes are integer indices, trans returns next-code logits.
        cb = ae_model.quantizer._embedding.weight.detach()  # (codebook_size, embedding_dim)
        codes_t = torch.from_numpy(code_arr).long().to(args.device)
        for i in range(0, len(z_arr), bs):
            c_b = codes_t[i:i+bs]            # (B, L)
            B = c_b.shape[0]
            c_rep = c_b.unsqueeze(1).repeat(1, n_actions, 1).view(B * n_actions, -1)  # (B*A, L)
            a_rep = torch.arange(n_actions, device=args.device).unsqueeze(0).repeat(B, 1).view(-1)
            with torch.no_grad():
                logits, r, g = trans_model(c_rep, a_rep, return_logits=True)
                next_codes = logits.argmax(dim=1)        # (B*A, L)
                # Translate codes to continuous embeddings, flatten to (B*A, L*emb_dim)
                z_next = cb[next_codes].permute(0, 2, 1).reshape(B * n_actions, L * embedding_dim)
            Tza[i:i+bs]   = z_next.view(B, n_actions, -1).cpu().numpy()
            R[i:i+bs]     = (r if r.dim() <= 2 else r.view(B * n_actions, -1)[:, 0]).view(B, n_actions).cpu().numpy()
            Gamma[i:i+bs] = (g if g.dim() <= 2 else g.view(B * n_actions, -1)[:, 0]).view(B, n_actions).cpu().numpy()
    else:
        z_t = torch.from_numpy(z_arr).to(args.device)
        for i in range(0, len(z_arr), bs):
            z_b = z_t[i:i+bs]
            B = z_b.shape[0]
            z_rep = z_b.unsqueeze(1).repeat(1, n_actions, 1).view(B * n_actions, -1)
            a_rep = torch.arange(n_actions, device=args.device).unsqueeze(0).repeat(B, 1).view(-1)
            with torch.no_grad():
                zn, r, g = trans_model(z_rep, a_rep)
                zn_flat = zn.view(B * n_actions, -1)
            Tza[i:i+bs]   = zn_flat.view(B, n_actions, -1).cpu().numpy()
            R[i:i+bs]     = r.view(B, n_actions).cpu().numpy()
            Gamma[i:i+bs] = g.view(B, n_actions).cpu().numpy()

    # state_change_norm: ||T(z, a) - z||   (avg over a, z)
    z_broadcast = z_arr[:, None, :]  # (N, 1, D)
    state_change = np.linalg.norm(Tza - z_broadcast, axis=2)  # (N, A)
    state_change_norm = float(state_change.mean())

    # action_diff_norm: ||T(z, a) - T(z, a')|| over all (a≠a') pairs
    pair_dists = []
    for a in range(n_actions):
        for ap in range(a + 1, n_actions):
            d = np.linalg.norm(Tza[:, a] - Tza[:, ap], axis=1)  # (N,)
            pair_dists.append(d)
    pair_dists = np.concatenate(pair_dists)
    action_diff_norm = float(pair_dists.mean())

    action_dep_ratio = action_diff_norm / max(state_change_norm, 1e-9)

    # Reward / gamma action-variance
    reward_action_var = float(R.var(axis=1).mean())
    gamma_action_var  = float(Gamma.var(axis=1).mean())
    reward_action_range = float((R.max(axis=1) - R.min(axis=1)).mean())
    gamma_action_range  = float((Gamma.max(axis=1) - Gamma.min(axis=1)).mean())

    # Effective action rank: SVD on the per-state matrix [T(z,a)-z for all a]
    # Stack action-deltas per state; rank tells us how many distinct directions
    # the WM can predict from one state.
    deltas = Tza - z_broadcast  # (N, A, D)
    # Subsample for SVD speed
    sub = np.random.choice(len(deltas), size=min(500, len(deltas)), replace=False)
    eff_ranks = []
    for i in sub:
        m = deltas[i]  # (A, D)
        # Tolerance scaled to state-change norm so a 1% direction counts as
        # informative; below that it's noise.
        tol = 0.01 * state_change[i].mean()
        s = np.linalg.svd(m, compute_uv=False)
        eff_ranks.append(int((s > tol).sum()))
    effective_action_rank = float(np.mean(eff_ranks))

    print(f'\n=== Phase N: action-conditioning probe ===')
    print(f'  state_change_norm    = {state_change_norm:.4f}   ||T(z,a) - z||')
    print(f'  action_diff_norm     = {action_diff_norm:.4f}   ||T(z,a) - T(z,a_other)||')
    print(f'  action_dep_ratio     = {action_dep_ratio:.4f}   (~0 ⇒ ignored, ≥1 ⇒ strong)')
    print(f'  reward_action_var    = {reward_action_var:.6f}')
    print(f'  gamma_action_var     = {gamma_action_var:.6f}')
    print(f'  reward_action_range  = {reward_action_range:.4f}')
    print(f'  gamma_action_range   = {gamma_action_range:.4f}')
    print(f'  effective_action_rank= {effective_action_rank:.2f}  / {n_actions} (max)')

    result = {
        'model_path': args.model_path,
        'env_name': margs.env_name,
        'n_actions': int(n_actions),
        'n_states': int(len(z_arr)),
        'state_change_norm': state_change_norm,
        'action_diff_norm': action_diff_norm,
        'action_dep_ratio': action_dep_ratio,
        'reward_action_var': reward_action_var,
        'gamma_action_var': gamma_action_var,
        'reward_action_range': reward_action_range,
        'gamma_action_range': gamma_action_range,
        'effective_action_rank': effective_action_rank,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved {args.output_json}')


if __name__ == '__main__':
    main()
