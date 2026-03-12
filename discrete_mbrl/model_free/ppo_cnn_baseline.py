"""
PPO CnnPolicy baseline using Stable-Baselines3.

Environment : MiniGrid-LavaCrossingS9N1-v0
Obs         : 72x72x3 RGB pixels (RGBImgObsWrapper -> ImgObsWrapper)
Policy      : CnnPolicy (SB3 NatureCNN feature extractor)
Parallel    : 8 x DummyVecEnv
Total steps : 300,000
Eval freq   : 30,000 steps, 20 episodes
"""

import os
import argparse
import numpy as np
import gymnasium as gym

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


ENV_NAME        = "MiniGrid-LavaCrossingS9N1-v0"
N_ENVS          = 8
TOTAL_STEPS     = 300_000
EVAL_FREQ       = 30_000      # per-env steps between evals
N_EVAL_EPISODES = 20
RUN_NAME        = "ppo_cnn_baseline"


def make_env(seed=0):
    def _init():
        env = gym.make(ENV_NAME)
        env = RGBImgObsWrapper(env)   # dict obs -> adds RGB 'image' key (72x72x3)
        env = ImgObsWrapper(env)      # dict -> Box(72,72,3)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",    type=str, default=RUN_NAME)
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    save_dir = f"./models/{ENV_NAME}"
    log_dir  = f"./logs/{args.run_name}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # Training envs
    train_env = DummyVecEnv([make_env(seed=args.seed + i) for i in range(N_ENVS)])
    train_env = VecTransposeImage(train_env)   # (H,W,C) -> (C,H,W)

    # Eval env (single)
    eval_env = DummyVecEnv([make_env(seed=args.seed + 1000)])
    eval_env = VecTransposeImage(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/{args.run_name}_best",
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        tensorboard_log=log_dir,
    )

    print(f"\n{'='*60}")
    print(f"  PPO CnnPolicy Baseline")
    print(f"  Env       : {ENV_NAME}")
    print(f"  Obs shape : 72x72x3 RGB -> (3,72,72) for CNN")
    print(f"  N envs    : {N_ENVS}")
    print(f"  Steps     : {args.total_steps:,}")
    print(f"  Eval freq : every {EVAL_FREQ:,} per-env steps")
    print(f"  Save dir  : {save_dir}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save final model
    final_path = f"{save_dir}/{args.run_name}_final"
    model.save(final_path)
    print(f"\nSaved final model to {final_path}.zip")

    # Final eval
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )
    print(f"\nTraining Complete!")
    print(f"  Final eval mean reward : {mean_reward:.4f} +/- {std_reward:.4f}")
    print(f"  Best model             : {save_dir}/{args.run_name}_best/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
