#!/usr/bin/env python3
"""
Evaluate a trained model-free PPO agent with VQVAE encoder.
Reports mean reward, episode length, success rate over N episodes.
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from torch.distributions import Categorical
from tqdm import tqdm

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model
from training_helpers import freeze_model, make_argparser, process_args
from model_free.rl_utils import interpret_layer_sizes
from shared.models import mlp


def strip_compiled_prefix(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_checkpoint(model_path, encoder, args, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if args.ae_model_type == 'vqvae':
        input_dim = args.embedding_dim * encoder.n_latent_embeds
    else:
        input_dim = encoder.latent_dim

    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_dim = env.action_space.n
    env.close()

    policy_hidden = interpret_layer_sizes(getattr(args, 'policy_hidden', [256, 256]))
    critic_hidden = interpret_layer_sizes(getattr(args, 'critic_hidden', [256, 256]))

    mlp_kwargs = {
        'activation': getattr(args, 'rl_activation', 'relu'),
        'discrete_input': args.ae_model_type == 'vqvae',
    }
    if args.ae_model_type == 'vqvae':
        mlp_kwargs['n_embeds'] = args.codebook_size
        mlp_kwargs['embed_dim'] = args.embedding_dim

    policy = mlp([input_dim] + policy_hidden + [act_dim], **mlp_kwargs)
    critic = mlp([input_dim] + critic_hidden + [1], **mlp_kwargs)

    policy.load_state_dict(strip_compiled_prefix(checkpoint['policy_state_dict']))
    critic.load_state_dict(strip_compiled_prefix(checkpoint['critic_state_dict']))

    # Optionally load encoder weights from checkpoint too
    # encoder.load_state_dict(strip_compiled_prefix(checkpoint['ae_model_state_dict']), strict=False)

    return policy.to(device).eval(), critic.to(device).eval(), checkpoint


def evaluate(env, encoder, policy, device, n_episodes=100,
             max_steps=1000, deterministic=True):
    results = defaultdict(list)

    for ep in tqdm(range(n_episodes), desc='Evaluating'):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        done = False
        total_reward = 0.0
        step = 0
        reached_goal = False

        while not done and step < max_steps:
            obs_tensor = preprocess_obs([obs])
            with torch.no_grad():
                state = encoder.encode(obs_tensor.to(device), return_one_hot=True).squeeze(0)
                logits = policy(state.unsqueeze(0))

                if deterministic:
                    action = logits.argmax(dim=-1).item()
                else:
                    action = Categorical(logits=logits).sample().item()

            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            total_reward += float(reward)
            step += 1

            # MiniGrid: positive reward typically means goal reached
            if float(reward) > 0:
                reached_goal = True

        results['reward'].append(total_reward)
        results['length'].append(step)
        results['success'].append(float(reached_goal))

    return results


def main():
    parser = make_argparser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--deterministic', action='store_true', default=True)
    # --stochastic is already defined by make_argparser() via add_model_args
    parser.add_argument('--policy_hidden', nargs='*', default=[256, 256])
    parser.add_argument('--critic_hidden', nargs='*', default=[256, 256])
    parser.add_argument('--rl_activation', default='relu')
    parser.add_argument('--stochastic_eval', action='store_true',
                        help='Use stochastic actions during evaluation')
    args = parser.parse_args()
    args = process_args(args)
    deterministic = not args.stochastic_eval

    # Environment
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    sample_obs = env.reset()
    sample_obs = sample_obs[0] if isinstance(sample_obs, tuple) else sample_obs
    sample_obs = preprocess_obs([sample_obs])

    # Encoder
    encoder, _ = construct_ae_model(sample_obs.shape[1:], args, load=True)
    encoder = encoder.to(args.device)
    freeze_model(encoder)
    encoder.eval()

    # Policy
    policy, critic, ckpt = load_checkpoint(
        args.model_path, encoder, args, args.device)

    print(f"\n{'='*60}")
    print(f"Evaluating: {args.env_name}")
    print(f"Checkpoint: {args.model_path}")
    print(f"  Training step: {ckpt.get('step', '?')}")
    print(f"  Training avg_reward: {ckpt.get('avg_reward', '?')}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Episodes: {args.n_episodes}")
    print(f"{'='*60}\n")

    # Run evaluation
    results = evaluate(
        env, encoder, policy, args.device,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        deterministic=deterministic)

    # Print summary
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Reward:  {np.mean(results['reward']):.4f} ± {np.std(results['reward']):.4f}"
          f"  (min={np.min(results['reward']):.4f}, max={np.max(results['reward']):.4f})")
    print(f"  Length:  {np.mean(results['length']):.1f} ± {np.std(results['length']):.1f}"
          f"  (min={np.min(results['length'])}, max={np.max(results['length'])})")
    print(f"  Success: {np.mean(results['success'])*100:.1f}%"
          f"  ({int(np.sum(results['success']))}/{args.n_episodes})")

    env.close()


if __name__ == '__main__':
    main()