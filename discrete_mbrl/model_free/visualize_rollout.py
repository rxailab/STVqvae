"""
Visualize rollouts of the trained VAE+PPO model.
Saves one GIF per episode to ./rollouts/

Usage:
  python visualize_rollout.py
  python visualize_rollout.py --model_path ./models/MiniGrid-LavaCrossingS9N1-v0/best_model.pt --n_episodes 5
"""

import argparse
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import imageio
import numpy as np
import torch
from argparse import Namespace
from torch.distributions import Categorical

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model
from shared.models import mlp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./models/MiniGrid-LavaCrossingS9N1-v0/best_model.pt')
    parser.add_argument('--n_episodes', '-n', type=int, default=5)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--tile_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--out_dir', type=str, default='./rollouts')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def load_checkpoint(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    saved_args = ckpt['args']
    args = Namespace(**saved_args)
    args.device = device
    args.load = True
    args.model_dir = os.path.dirname(os.path.dirname(model_path))  # models/../
    return ckpt, args


def _strip_compile_prefix(state_dict):
    """Strip _orig_mod. prefix added by torch.compile before saving."""
    if any(k.startswith('_orig_mod.') for k in state_dict):
        return {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def build_models(ckpt, args, sample_obs, act_dim):
    # Reconstruct encoder
    ae_model, _ = construct_ae_model(
        sample_obs.shape[1:], args, latent_activation=True, load=False
    )
    ae_model.load_state_dict(_strip_compile_prefix(ckpt['ae_model_state_dict']))
    ae_model.to(args.device)
    ae_model.eval()

    # Mirror train.py logic: e2e VQVAE uses continuous quantized embeddings, not one-hot
    vqvae_e2e = (args.ae_model_type == 'vqvae' and getattr(args, 'e2e_loss', False))

    mlp_kwargs = {
        'activation': args.rl_activation,
        'discrete_input': (args.ae_model_type == 'vqvae') and (not vqvae_e2e),
    }
    if args.ae_model_type == 'vqvae':
        input_dim = args.embedding_dim * ae_model.n_latent_embeds
        if not vqvae_e2e:
            mlp_kwargs['n_embeds'] = args.codebook_size
            mlp_kwargs['embed_dim'] = args.embedding_dim
    else:
        input_dim = ae_model.latent_dim

    policy = mlp([input_dim] + args.policy_hidden + [act_dim], **mlp_kwargs)
    policy.load_state_dict(_strip_compile_prefix(ckpt['policy_state_dict']))
    policy.to(args.device)
    policy.eval()

    return ae_model, policy, vqvae_e2e


def get_frame(env, tile_size):
    return env.unwrapped.get_full_render(highlight=True, tile_size=tile_size)


def run_episode(ae_model, policy, env, tile_size, deterministic, device, vqvae_e2e=False):
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    frames = [get_frame(env, tile_size)]
    total_reward = 0.0
    done = False

    while not done:
        obs_preprocessed = preprocess_obs([obs])

        with torch.no_grad():
            state = ae_model.encode(
                obs_preprocessed.to(device),
                return_one_hot=not vqvae_e2e,
                return_quantized=vqvae_e2e,
            )
            logits = policy(state)
            if deterministic:
                action = int(logits.argmax(dim=-1).item())
            else:
                action = int(Categorical(logits=logits).sample().item())

        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            obs, reward, done, _ = step_result

        total_reward += reward
        frames.append(get_frame(env, tile_size))

    return frames, total_reward, len(frames) - 1


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'No model found at {args.model_path}')

    print(f'Loading model from {args.model_path}')
    ckpt, model_args = load_checkpoint(args.model_path, args.device)
    print(f'  Checkpoint step: {ckpt["step"]:,}  |  avg_reward: {ckpt["avg_reward"]:.4f}')

    env = make_env(model_args.env_name, max_steps=args.max_steps)
    act_dim = env.action_space.n

    reset_result = env.reset()
    sample_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs_t = preprocess_obs([sample_obs])

    ae_model, policy, vqvae_e2e = build_models(ckpt, model_args, sample_obs_t, act_dim)
    print(f'  AE params: {sum(p.numel() for p in ae_model.parameters()):,}')
    print(f'  Policy params: {sum(p.numel() for p in policy.parameters()):,}')
    print(f'  vqvae_e2e: {vqvae_e2e}')

    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(args.n_episodes):
        env = make_env(model_args.env_name, max_steps=args.max_steps)
        frames, total_reward, n_steps = run_episode(
            ae_model, policy, env, args.tile_size, args.deterministic, args.device,
            vqvae_e2e=vqvae_e2e,
        )
        env.close()

        status = 'SUCCESS' if total_reward > 0 else 'FAILED'
        gif_path = os.path.join(
            args.out_dir,
            f'ep{ep+1:02d}_{status}_r{total_reward:.2f}.gif'
        )
        imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)
        print(f'  Episode {ep+1}: {status} | reward={total_reward:.2f} | '
              f'steps={n_steps} | saved -> {gif_path}')
