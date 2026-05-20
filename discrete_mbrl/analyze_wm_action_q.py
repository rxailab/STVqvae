"""
Phase M: action-Q ranking under the WM.

Phase K showed cross-episode return ranking is broken even at K=1: the WM
cannot rank realized G_real from a one-step bootstrap. This analyzer tests
whether the WM can do something a planner *actually* needs: rank ACTIONS at
a fixed state.

For each visited state z, we compute the WM's one-step Q-estimate for every
discrete action a:
    Q̂(z, a) = r̂(z, a) + γ̂(z, a) · V(T(z, a))

Metrics (per checkpoint):
  - top1_agreement      : P(argmax_a Q̂(z, a) == policy(z))
  - greedy_improvement  : G_real under the policy that picks argmax_a Q̂ vs
                          G_real under the learned policy (rolled out in env)
  - q_margin            : mean(Q̂_max - Q̂_second)
  - q_action_var        : mean(Var_a(Q̂(z, a))) — does the WM differentiate
                          actions at all?

Interpretation:
  - top1 high (>0.7) ⇒ WM-induced policy is consistent with the trained policy
    (sanity check that the WM at least respects the policy's action ordering).
  - greedy_improvement > 0 ⇒ the WM can improve the policy. This is the
    actor-critic use case (Dreamer / MuZero) and would be a positive result
    sufficient to flip the paper from "negative" to "diagnosis-and-fix".
  - q_action_var ≈ 0 ⇒ the WM gives the same Q for every action, meaning it
    can't be used for action selection at all.

Usage:
    python analyze_wm_action_q.py --model_path ckpt.pt --output_json out.json
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

from env_helpers import make_env, preprocess_obs  # noqa: E402
from analyze_wm_multistep import load_checkpoint  # noqa: E402
from analyze_wm_phaseB_e8 import build_mlp, encode_flat  # noqa: E402


def all_action_q(trans_model, ae_model, critic, z, n_actions, device, trans_kind):
    """For a single latent state z (shape (D,)), return Q̂(z, a) for all actions.
    Builds a batch of size n_actions where the same state is repeated."""
    if trans_kind == 'discrete':
        # Skip discrete for this analyzer (codebook input not directly available
        # from a flat z). Phase M targets continuous WMs only.
        return None
    z_batch = torch.from_numpy(z).unsqueeze(0).repeat(n_actions, 1).to(device)
    a_batch = torch.arange(n_actions, dtype=torch.long, device=device)
    with torch.no_grad():
        z_next, r_pred, g_pred = trans_model(z_batch, a_batch)
        z_next_flat = z_next.view(n_actions, -1)
        v_next = critic(z_next_flat).squeeze(-1)
        Q = r_pred.squeeze(-1) + g_pred.squeeze(-1) * v_next
    return Q.cpu().numpy()  # (n_actions,)


def collect_state_action_dataset(env, ae_model, policy, n_steps, device, trans_kind):
    """Run the policy and record (z, a_policy) pairs from real states."""
    z_list, a_list = [], []
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    obs_t = torch.from_numpy(obs).float()
    for t in range(n_steps):
        z = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
        with torch.no_grad():
            logits = policy(torch.from_numpy(z).unsqueeze(0).to(device))
            probs = F.softmax(logits.squeeze(0), dim=-1)
            a = int(torch.multinomial(probs, 1).item())
        z_list.append(z.astype(np.float32))
        a_list.append(a)
        sr = env.step(a)
        obs_next = sr[0] if isinstance(sr, tuple) else sr
        done = sr[2] if len(sr) >= 3 else False
        obs_t = torch.from_numpy(obs_next).float()
        if done:
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
            obs_t = torch.from_numpy(obs).float()
    return np.stack(z_list), np.array(a_list)


def rollout_with_wm_greedy_policy(env, ae_model, trans_model, critic, n_actions,
                                  n_episodes, max_steps, device, trans_kind):
    """Roll out a policy that, at each step, picks argmax_a Q̂(z, a) under the WM.
    Returns list of episode returns under this WM-greedy policy."""
    returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        ep_r = 0.0
        for t in range(max_steps):
            z = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
            Q = all_action_q(trans_model, ae_model, critic, z, n_actions, device, trans_kind)
            if Q is None:
                return None
            a = int(np.argmax(Q))
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            r = sr[1] if len(sr) >= 2 else 0.0
            done = sr[2] if len(sr) >= 3 else False
            ep_r += float(r)
            obs_t = torch.from_numpy(obs_next).float()
            if done:
                break
        returns.append(ep_r)
    return returns


def rollout_policy(env, ae_model, policy, n_episodes, max_steps, device, trans_kind, stochastic=False):
    """Roll out the learned policy for return baseline."""
    returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        ep_r = 0.0
        for t in range(max_steps):
            z = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
            with torch.no_grad():
                logits = policy(torch.from_numpy(z).unsqueeze(0).to(device))
                if stochastic:
                    probs = F.softmax(logits.squeeze(0), dim=-1)
                    a = int(torch.multinomial(probs, 1).item())
                else:
                    a = int(logits.argmax(dim=-1).item())
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            r = sr[1] if len(sr) >= 2 else 0.0
            done = sr[2] if len(sr) >= 3 else False
            ep_r += float(r)
            obs_t = torch.from_numpy(obs_next).float()
            if done:
                break
        returns.append(ep_r)
    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_states', type=int, default=4000,
                        help='Number of policy states to score for top-1 / margin metrics.')
    parser.add_argument('--n_eval_eps', type=int, default=100,
                        help='Episodes for greedy-improvement evaluation.')
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()

    print(f'Loading {args.model_path}')
    ae_model, trans_model, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_model is None:
        raise RuntimeError('No transition model.')
    if trans_kind == 'discrete':
        print('[skip] discrete WM; Phase M targets continuous WMs only.')
        return

    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    input_dim = L * embedding_dim
    n_actions = env.action_space.n

    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    policy = build_mlp(input_dim, src['args'].get('policy_hidden', [256, 256]), n_actions, 'relu').to(args.device).eval()
    critic = build_mlp(input_dim, src['args'].get('critic_hidden', [256, 256]), 1, 'relu').to(args.device).eval()
    policy.load_state_dict(src['policy_state_dict'])
    critic.load_state_dict(src['critic_state_dict'])

    # ------ [1/3] State dataset + per-state Q̂ over actions ------
    print(f'\n[1/3] Collecting {args.n_states} policy states for action-Q analysis')
    z_arr, a_pol_arr = collect_state_action_dataset(env, ae_model, policy, args.n_states,
                                                     args.device, trans_kind)

    print('  computing Q̂(z, a) for all actions across all states...')
    Q_all = np.zeros((len(z_arr), n_actions), dtype=np.float32)
    for i, z in enumerate(z_arr):
        Q_all[i] = all_action_q(trans_model, ae_model, critic, z, n_actions, args.device, trans_kind)

    a_wm = Q_all.argmax(axis=1)
    top1_agreement = float((a_wm == a_pol_arr).mean())
    Q_sorted = np.sort(Q_all, axis=1)
    q_margin = float((Q_sorted[:, -1] - Q_sorted[:, -2]).mean())
    q_action_var = float(Q_all.var(axis=1).mean())
    q_range = float((Q_all.max(axis=1) - Q_all.min(axis=1)).mean())

    # Per-action histogram of agreement
    per_action = {}
    for a in range(n_actions):
        mask = (a_pol_arr == a)
        if mask.sum() == 0:
            per_action[a] = {'n': 0, 'agreement': None}
        else:
            per_action[a] = {'n': int(mask.sum()),
                             'agreement': float((a_wm[mask] == a).mean())}

    print(f'  top1_agreement(WM, policy) = {top1_agreement:.3f}')
    print(f'  q_margin(top1 - top2)      = {q_margin:+.4f}')
    print(f'  q_action_var               = {q_action_var:+.6f}')
    print(f'  q_range(max - min over a)  = {q_range:+.4f}')

    # ------ [2/3] Roll out the WM-greedy policy in the real env ------
    print(f'\n[2/3] Evaluating WM-greedy policy in real env ({args.n_eval_eps} eps)')
    wm_greedy_returns = rollout_with_wm_greedy_policy(env, ae_model, trans_model, critic,
                                                      n_actions, args.n_eval_eps,
                                                      args.max_steps, args.device, trans_kind)
    print(f'  WM-greedy:    mean return = {np.mean(wm_greedy_returns):+.4f}  '
          f'std={np.std(wm_greedy_returns):.4f}  succ={np.mean(np.array(wm_greedy_returns) > 0):.0%}')

    # ------ [3/3] Roll out the learned policy (det + stochastic) for baseline ------
    print(f'\n[3/3] Evaluating learned policy ({args.n_eval_eps} eps each)')
    pol_det = rollout_policy(env, ae_model, policy, args.n_eval_eps, args.max_steps, args.device, trans_kind, stochastic=False)
    pol_stoch = rollout_policy(env, ae_model, policy, args.n_eval_eps, args.max_steps, args.device, trans_kind, stochastic=True)
    print(f'  policy det:   mean return = {np.mean(pol_det):+.4f}  succ={np.mean(np.array(pol_det) > 0):.0%}')
    print(f'  policy stoch: mean return = {np.mean(pol_stoch):+.4f}  succ={np.mean(np.array(pol_stoch) > 0):.0%}')

    greedy_improvement = float(np.mean(wm_greedy_returns) - np.mean(pol_det))

    print('\n=== Phase M: action-Q ranking summary ===')
    print(f'  top1_agreement     = {top1_agreement:.3f}')
    print(f'  q_margin           = {q_margin:+.4f}')
    print(f'  q_action_var       = {q_action_var:+.6f}')
    print(f'  WM-greedy return   = {np.mean(wm_greedy_returns):+.4f}')
    print(f'  policy-det return  = {np.mean(pol_det):+.4f}')
    print(f'  greedy_improvement = {greedy_improvement:+.4f}  ({"+" if greedy_improvement > 0 else "-"}WM helps)')

    result = {
        'model_path': args.model_path,
        'env_name': margs.env_name,
        'n_actions': int(n_actions),
        'n_states': int(len(z_arr)),
        'n_eval_eps': args.n_eval_eps,
        'top1_agreement_wm_vs_policy': top1_agreement,
        'q_margin_top1_minus_top2_mean': q_margin,
        'q_action_var_mean': q_action_var,
        'q_range_mean': q_range,
        'per_action': {str(k): v for k, v in per_action.items()},
        'wm_greedy_return_mean': float(np.mean(wm_greedy_returns)),
        'wm_greedy_return_std':  float(np.std(wm_greedy_returns)),
        'wm_greedy_success_rate': float(np.mean(np.array(wm_greedy_returns) > 0)),
        'policy_det_return_mean': float(np.mean(pol_det)),
        'policy_det_success_rate': float(np.mean(np.array(pol_det) > 0)),
        'policy_stoch_return_mean': float(np.mean(pol_stoch)),
        'policy_stoch_success_rate': float(np.mean(np.array(pol_stoch) > 0)),
        'greedy_improvement': greedy_improvement,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved {args.output_json}')


if __name__ == '__main__':
    main()
