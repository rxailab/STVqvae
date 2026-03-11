"""
Visualize rollouts of the trained PPO baseline on a MiniGrid environment.
Saves one GIF per episode to the current directory.

Usage:
  python visualize_rollout.py
  python visualize_rollout.py --env_name MiniGrid-LavaCrossingS9N1-v0 --goal_type goal --n_episodes 5
"""

import argparse
import os
import sys
sys.path.append('../')

import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy

from env_helpers import make_env
from policies import MODEL_SAVE_FORMAT
from wrappers import SB3ObsWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', type=str,
                        default='MiniGrid-LavaCrossingS9N1-v0')
    parser.add_argument('--goal_type', '-g', type=str, default='goal')
    parser.add_argument('--n_episodes', '-n', type=int, default=3)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--tile_size', type=int, default=32,
                        help='Tile size for rendered frames (larger = higher res)')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--out_dir', type=str, default='./rollouts')
    return parser.parse_args()


def make_obs_env(env_name, max_steps):
    """Create env with same wrapper chain used during training."""
    env = make_env(env_name, max_steps=max_steps)
    env = SB3ObsWrapper(env)
    return env


def get_frame(env, tile_size):
    """Render a high-res RGB frame from the unwrapped MiniGrid env."""
    return env.unwrapped.get_full_render(highlight=True, tile_size=tile_size)


def run_episode(model, env, tile_size, deterministic):
    """Run one episode and return (frames, total_reward, n_steps)."""
    obs, _ = env.reset()
    frames = [get_frame(env, tile_size)]
    total_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        frames.append(get_frame(env, tile_size))
        done = terminated or truncated

    return frames, total_reward, len(frames) - 1


if __name__ == '__main__':
    args = parse_args()

    model_path = MODEL_SAVE_FORMAT.format(args.env_name, args.goal_type)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No model found at {model_path}')

    print(f'Loading model from {model_path}')
    model = ActorCriticPolicy.load(model_path)

    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(args.n_episodes):
        env = make_obs_env(args.env_name, args.max_steps)
        frames, total_reward, n_steps = run_episode(
            model, env, args.tile_size, args.deterministic)
        env.close()

        success = total_reward > 0
        status = 'SUCCESS' if success else 'FAILED'
        gif_path = os.path.join(
            args.out_dir,
            f'ep{ep+1:02d}_{status}_r{total_reward:.2f}.gif')
        imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)
        print(f'  Episode {ep+1}: {status} | reward={total_reward:.2f} | '
              f'steps={n_steps} | saved -> {gif_path}')
