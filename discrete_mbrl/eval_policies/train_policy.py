# python train_policy.py --env_name X --goal_type Y
# Get the env
# Create a list of different reward wrappers
# Apply the reward wrappers to the env based on the goal type
# Create the policy with SB3
# Train the policy, and save it to the models directory

import argparse
import os
import sys
sys.path.append('../')

from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv


from env_helpers import make_env
from policies import MODEL_SAVE_FORMAT
from wrappers import SB3ObsWrapper, apply_goal_wrapper


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', '-e', type=str, required=True)
  parser.add_argument('--goal_type', '-g', type=str, required=True)
  parser.add_argument('--n_envs', '-n', type=int, default=8)
  parser.add_argument('--train_steps', '-s', type=int, default=int(3e5))
  parser.add_argument('--eval_freq', type=int, default=30_000)
  parser.add_argument('--eval_episodes', type=int, default=20)
  parser.add_argument('--policy_model', type=str, default='CnnPolicy')
  parser.add_argument('--n_steps', type=int, default=128,
                      help='PPO rollout steps per env per update. Default 128 gives '
                           '~292 updates for 300k steps with 8 envs (vs 18 with SB3 '
                           'default of 2048).')
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--n_epochs', type=int, default=10)
  parser.add_argument('--learning_rate', type=float, default=3e-4)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--gae_lambda', type=float, default=0.95)
  parser.add_argument('--clip_range', type=float, default=0.2)
  parser.add_argument('--ent_coef', type=float, default=0.0)
  parser.add_argument('--vf_coef', type=float, default=0.5)
  parser.add_argument('--run_name', type=str, default=None)
  return parser.parse_args()


def make_single_env(env_name, goal_type):
  env = make_env(env_name)
  env = apply_goal_wrapper(env, goal_type)
  env = SB3ObsWrapper(env)
  env = RecordEpisodeStatistics(env)
  return env


if __name__ == '__main__':
  args = parse_args()

  # Create vectorized training env (DummyVecEnv works on all platforms)
  env = DummyVecEnv([
    (lambda: make_single_env(args.env_name, args.goal_type))
    for _ in range(args.n_envs)
  ])

  # Separate eval env
  eval_env = DummyVecEnv([lambda: make_single_env(args.env_name, args.goal_type)])

  # Create callback to save highest reward model
  eval_callback = EvalCallback(
    eval_env, n_eval_episodes=args.eval_episodes, eval_freq=args.eval_freq,
    best_model_save_path='./logs/', log_path='./logs/',
    deterministic=True)

  n_updates = args.train_steps // (args.n_steps * args.n_envs)
  print(f'PPO setup: n_steps={args.n_steps}, n_envs={args.n_envs}, '
        f'total_steps={args.train_steps:,} -> ~{n_updates} updates')

  # Create and train policy
  model = PPO(
    args.policy_model, env, verbose=1, tensorboard_log='./logs/',
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    learning_rate=args.learning_rate,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
    ent_coef=args.ent_coef,
    vf_coef=args.vf_coef,
  )
  model.learn(total_timesteps=args.train_steps, callback=eval_callback)

  # Save the final trained policy
  run_tag = args.run_name or f'{args.env_name}_{args.goal_type}'
  save_path = MODEL_SAVE_FORMAT.format(args.env_name, args.goal_type)
  save_dir = os.path.dirname(save_path)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  final_path = os.path.join(save_dir, f'{run_tag}_final')
  model.save(final_path)
  print(f'Saved final policy to {final_path}.zip')

  # Print eval stats from the best eval checkpoint
  eval_stats = np.load('./logs/evaluations.npz')
  rewards = eval_stats['results']
  ep_lens = eval_stats['ep_lengths']
  mean_rewards = rewards.mean(axis=1)
  best_idx = np.argmax(mean_rewards)
  print('Best Eval Checkpoint Stats:')
  print(f'Timestep: {eval_stats["timesteps"][best_idx]}')
  print('Reward mean: {:.2f} | std: {:.2f} | min: {:.2f} | max: {:.2f}'.format(
    mean_rewards[best_idx], rewards[best_idx].std(),
    rewards[best_idx].min(), rewards[best_idx].max()))
  print('Episode length mean: {:.2f} | std: {:.2f} | min: {:.2f} | max: {:.2f}'.format(
    ep_lens[best_idx].mean(), ep_lens[best_idx].std(),
    ep_lens[best_idx].min(), ep_lens[best_idx].max()))

  # Delete the logged data (best-effort cleanup)
  for f in ('./logs/best_model.zip', './logs/evaluations.npz'):
    try:
      os.remove(f)
    except OSError:
      pass