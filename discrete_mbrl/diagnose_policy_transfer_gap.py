#!/usr/bin/env python3
import argparse
import json
import os
import sys
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env_helpers import make_env, preprocess_act, preprocess_obs
from model_construction import construct_ae_model, construct_trans_model
from train_rl_model import (
    GymnasiumCompatWrapper,
    ObsEncoderWrapperGymnasium,
    PredictedModelWrapperGymnasium,
    rl_eval,
)

GAMMA_CONST = 0.99


def build_args(namespace):
    args = vars(namespace).copy()
    args.setdefault("wandb", False)
    args.setdefault("comet_ml", False)
    args.setdefault("ae_grad_clip", 0.0)
    args.setdefault("env_max_steps", None)
    args.setdefault("stochastic", None)
    args.setdefault("trans_hidden", 256)
    args.setdefault("trans_depth", 3)
    args.setdefault("learning_rate", 3e-4)
    args.setdefault("trans_learning_rate", 3e-4)
    args.setdefault("codebook_size", None)
    args.setdefault("embedding_dim", 64)
    args.setdefault("e2e_loss", False)
    args.setdefault("log_norms", False)
    args.setdefault("n_trans_options", 32)
    args.setdefault("fta_tiles", 20)
    args.setdefault("fta_bound_low", -2.0)
    args.setdefault("fta_bound_high", 2.0)
    args.setdefault("fta_eta", 0.2)
    args.setdefault("repr_sparsity", 0.0)
    args.setdefault("sparsity_type", "l1")
    args.setdefault("commitment_cost", 0.25)
    args.setdefault("ema_decay", 0.99)
    return SimpleNamespace(**args)


def checkpoint_path(model_dir, env_name, model_hash):
    return os.path.join(model_dir, "models", env_name, f"model_{model_hash}.pt")


def load_encoder(args, env_name, device):
    env = make_env(env_name, max_steps=args.env_max_steps)
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([obs])
    env.close()

    ae_model_hash = args.ae_model_hash
    encoder = construct_ae_model(sample_obs.shape[1:], args, load=False)[0]
    args.ae_model_hash = ae_model_hash
    encoder.load_state_dict(torch.load(
        checkpoint_path(args.model_dir, env_name, ae_model_hash),
        map_location=device,
        weights_only=False,
    ))
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def load_transition_model(args, encoder, env_name, device):
    env = make_env(env_name, max_steps=args.env_max_steps)
    trans_model_hash = args.trans_model_hash
    trans_model = construct_trans_model(encoder, args, env.action_space, load=False)[0]
    args.trans_model_hash = trans_model_hash
    env.close()
    trans_model.load_state_dict(torch.load(
        checkpoint_path(args.model_dir, env_name, trans_model_hash),
        map_location=device,
        weights_only=False,
    ))
    trans_model = trans_model.to(device).eval()
    for param in trans_model.parameters():
        param.requires_grad = False
    return trans_model


def make_encoded_env(env_name, encoder, max_steps):
    def _factory():
        base_env = make_env(env_name, max_steps=max_steps)
        return Monitor(ObsEncoderWrapperGymnasium(base_env, encoder))

    return DummyVecEnv([_factory])


def make_world_model_eval_env(env_name, encoder, trans_model, real_max_steps, wm_horizon):
    def _factory():
        base_env = make_env(env_name, max_steps=real_max_steps)
        world_model = PredictedModelWrapperGymnasium(base_env, encoder, trans_model)
        capped_horizon = wm_horizon if wm_horizon > 0 else real_max_steps
        world_model = gymnasium.wrappers.TimeLimit(world_model, max_episode_steps=capped_horizon)
        return Monitor(world_model)

    return DummyVecEnv([_factory])


def encode_episode_latents(encoder, obs_seq, device):
    obs_tensor = preprocess_obs(obs_seq).to(device)
    with torch.no_grad():
        return encoder.encode(obs_tensor)


def collect_real_policy_episodes(encoder, policy, env_name, device, n_episodes, max_steps):
    encoded_env = make_encoded_env(env_name, encoder, max_steps)
    real_env = make_env(env_name, max_steps=max_steps)
    episodes = []

    for _ in tqdm(range(n_episodes), desc="Collecting real-policy episodes"):
        obs = encoded_env.reset()
        reset_result = real_env.reset()
        real_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs_seq = [real_obs]
        actions = []
        rewards = []
        dones = []

        done = False
        while not done and len(actions) < max_steps:
            action, _ = policy.predict(obs, deterministic=True)
            action_int = int(np.asarray(action).reshape(-1)[0])

            obs, _, done_arr, _ = encoded_env.step(action)
            step_result = real_env.step(action_int)
            if len(step_result) == 5:
                next_real_obs, real_reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                next_real_obs, real_reward, done, _ = step_result

            obs_seq.append(next_real_obs)
            actions.append(action_int)
            rewards.append(float(real_reward))
            dones.append(float(done))

            if isinstance(done_arr, np.ndarray):
                done = bool(done_arr.reshape(-1)[0]) or done

        episodes.append({
            "obs": obs_seq,
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        })

    encoded_env.close()
    real_env.close()
    return episodes


def evaluate_policy_in_env(policy, env, encoder, n_eval_episodes):
    mean_reward, std_reward, mean_len = rl_eval(
        policy, env, encoder, n_eval_episodes=n_eval_episodes, log=False
    )
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_episode_length": float(mean_len),
    }


def build_teacher_forced_batch(encoded_eps):
    curr_states = []
    actions = []
    next_states = []
    rewards = []
    gammas = []

    for episode in encoded_eps:
        z = episode["z"]
        for step_idx, action in enumerate(episode["actions"]):
            curr_states.append(z[step_idx])
            actions.append(action)
            next_states.append(z[step_idx + 1])
            rewards.append(episode["rewards"][step_idx])
            gammas.append(GAMMA_CONST * (1.0 - episode["dones"][step_idx]))

    if not curr_states:
        return None

    return {
        "curr_states": torch.stack(curr_states),
        "actions": torch.tensor(actions, dtype=torch.long),
        "next_states": torch.stack(next_states),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "gammas": torch.tensor(gammas, dtype=torch.float32),
    }


def evaluate_teacher_forced(trans_model, batch, device):
    curr_states = batch["curr_states"].to(device)
    actions = batch["actions"].to(device)
    next_states = batch["next_states"].to(device)
    rewards = batch["rewards"].to(device)
    gammas = batch["gammas"].to(device)

    with torch.no_grad():
        pred_states, pred_rewards, pred_gammas = trans_model(curr_states, actions)
        pred_rewards = pred_rewards.squeeze(-1)
        pred_gammas = pred_gammas.squeeze(-1)

    state_mse = torch.pow(pred_states - next_states, 2).view(pred_states.shape[0], -1).mean(dim=1)
    reward_abs = torch.abs(pred_rewards - rewards)
    reward_mse = torch.pow(pred_rewards - rewards, 2)
    gamma_abs = torch.abs(pred_gammas - gammas)
    gamma_mse = torch.pow(pred_gammas - gammas, 2)

    return {
        "segments": int(curr_states.shape[0]),
        "state_mse": float(state_mse.mean().item()),
        "reward_abs_error": float(reward_abs.mean().item()),
        "reward_mse": float(reward_mse.mean().item()),
        "gamma_abs_error": float(gamma_abs.mean().item()),
        "gamma_mse": float(gamma_mse.mean().item()),
        "pred_reward_mean": float(pred_rewards.mean().item()),
        "actual_reward_mean": float(rewards.mean().item()),
        "pred_gamma_mean": float(pred_gammas.mean().item()),
        "actual_gamma_mean": float(gammas.mean().item()),
    }


def build_horizon_batch(encoded_eps, horizon):
    start_states = []
    action_seqs = []
    target_states = []
    reward_seqs = []
    done_seqs = []

    for episode in encoded_eps:
        z = episode["z"]
        actions = episode["actions"]
        rewards = episode["rewards"]
        dones = episode["dones"]
        total_steps = len(actions)
        if total_steps < horizon:
            continue
        for start in range(total_steps - horizon + 1):
            end = start + horizon
            start_states.append(z[start])
            action_seqs.append(actions[start:end])
            target_states.append(z[end])
            reward_seqs.append(rewards[start:end])
            done_seqs.append(dones[start:end])

    if not start_states:
        return None

    return {
        "start_states": torch.stack(start_states),
        "action_seqs": torch.tensor(np.stack(action_seqs), dtype=torch.long),
        "target_states": torch.stack(target_states),
        "reward_seqs": torch.tensor(np.stack(reward_seqs), dtype=torch.float32),
        "done_seqs": torch.tensor(np.stack(done_seqs), dtype=torch.float32),
    }


def evaluate_open_loop(trans_model, batch, horizon, device):
    start_states = batch["start_states"].to(device)
    action_seqs = batch["action_seqs"].to(device)
    target_states = batch["target_states"].to(device)
    reward_seqs = batch["reward_seqs"].to(device)
    done_seqs = batch["done_seqs"].to(device)

    curr = start_states
    pred_rewards = []
    pred_gammas = []
    with torch.no_grad():
        for step_idx in range(horizon):
            curr, reward, gamma = trans_model(curr, action_seqs[:, step_idx])
            pred_rewards.append(reward.squeeze(-1))
            pred_gammas.append(gamma.squeeze(-1))

    pred_reward_seq = torch.stack(pred_rewards, dim=1)
    pred_gamma_seq = torch.stack(pred_gammas, dim=1)

    state_mse = torch.pow(curr - target_states, 2).view(curr.shape[0], -1).mean(dim=1)
    reward_abs = torch.abs(pred_reward_seq - reward_seqs).mean(dim=1)
    gamma_targets = GAMMA_CONST * (1.0 - done_seqs)
    gamma_abs = torch.abs(pred_gamma_seq - gamma_targets).mean(dim=1)

    actual_returns = []
    predicted_returns = []
    actual_alive = []
    predicted_alive = []
    for row_idx in range(action_seqs.shape[0]):
        actual_ret = 0.0
        pred_ret = 0.0
        actual_discount = 1.0
        pred_discount = 1.0
        actual_survival = 1.0
        pred_survival = 1.0
        for step_idx in range(horizon):
            actual_ret += actual_discount * float(reward_seqs[row_idx, step_idx].item())
            pred_ret += pred_discount * float(pred_reward_seq[row_idx, step_idx].item())

            actual_gamma = float(gamma_targets[row_idx, step_idx].item())
            pred_gamma = float(pred_gamma_seq[row_idx, step_idx].item())
            actual_survival *= actual_gamma
            pred_survival *= pred_gamma
            actual_discount *= actual_gamma
            pred_discount *= pred_gamma

        actual_returns.append(actual_ret)
        predicted_returns.append(pred_ret)
        actual_alive.append(actual_survival)
        predicted_alive.append(pred_survival)

    actual_returns = np.asarray(actual_returns, dtype=np.float64)
    predicted_returns = np.asarray(predicted_returns, dtype=np.float64)
    actual_alive = np.asarray(actual_alive, dtype=np.float64)
    predicted_alive = np.asarray(predicted_alive, dtype=np.float64)

    return {
        "segments": int(start_states.shape[0]),
        "open_loop_state_mse": float(state_mse.mean().item()),
        "step_reward_abs_error": float(reward_abs.mean().item()),
        "step_gamma_abs_error": float(gamma_abs.mean().item()),
        "discounted_return_actual_mean": float(actual_returns.mean()),
        "discounted_return_pred_mean": float(predicted_returns.mean()),
        "discounted_return_bias": float((predicted_returns - actual_returns).mean()),
        "discounted_return_abs_error": float(np.abs(predicted_returns - actual_returns).mean()),
        "continuation_actual_mean": float(actual_alive.mean()),
        "continuation_pred_mean": float(predicted_alive.mean()),
        "continuation_bias": float((predicted_alive - actual_alive).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ae_model_type", type=str, default="vae")
    parser.add_argument("--ae_model_version", type=str, default="2")
    parser.add_argument("--ae_model_hash", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--filter_size", type=int, default=9)
    parser.add_argument("--trans_model_type", type=str, default="continuous")
    parser.add_argument("--trans_model_version", type=str, default="1")
    parser.add_argument("--trans_model_hash", type=str, required=True)
    parser.add_argument("--trans_hidden", type=int, default=512)
    parser.add_argument("--trans_depth", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--n_diag_episodes", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 5, 8, 12, 16])
    parser.add_argument("--world_model_horizons", nargs="*", type=int, default=[8, 12, 16, 500])
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    args = build_args(args)
    args.device = device

    encoder = load_encoder(args, args.env_name, device)
    trans_model = load_transition_model(args, encoder, args.env_name, device)
    policy = PPO.load(args.policy_path, device=device)

    real_eval_env = make_encoded_env(args.env_name, encoder, args.max_steps)
    try:
        real_eval = evaluate_policy_in_env(policy, real_eval_env, encoder, args.n_eval_episodes)
    finally:
        real_eval_env.close()

    imagined_evals = {}
    for wm_horizon in args.world_model_horizons:
        env = make_world_model_eval_env(
            args.env_name, encoder, trans_model, args.max_steps, wm_horizon
        )
        try:
            key = str(wm_horizon if wm_horizon > 0 else args.max_steps)
            imagined_evals[key] = evaluate_policy_in_env(policy, env, encoder, args.n_eval_episodes)
        finally:
            env.close()

    episodes = collect_real_policy_episodes(
        encoder,
        policy,
        args.env_name,
        device,
        args.n_diag_episodes,
        args.max_steps,
    )
    encoded_eps = []
    for episode in tqdm(episodes, desc="Encoding trajectories"):
        encoded_eps.append({
            "z": encode_episode_latents(encoder, episode["obs"], device).cpu(),
            "actions": episode["actions"],
            "rewards": episode["rewards"],
            "dones": episode["dones"],
        })

    teacher_batch = build_teacher_forced_batch(encoded_eps)
    teacher_forced = evaluate_teacher_forced(trans_model, teacher_batch, device) if teacher_batch else {}

    horizons = {}
    for horizon in args.horizons:
        batch = build_horizon_batch(encoded_eps, horizon)
        horizons[str(horizon)] = (
            evaluate_open_loop(trans_model, batch, horizon, device)
            if batch is not None else {"segments": 0}
        )

    results = {
        "env_name": args.env_name,
        "policy_path": args.policy_path,
        "model_dir": args.model_dir,
        "ae_model_hash": args.ae_model_hash,
        "trans_model_hash": args.trans_model_hash,
        "device": device,
        "n_eval_episodes": args.n_eval_episodes,
        "n_diag_episodes": args.n_diag_episodes,
        "max_steps": args.max_steps,
        "real_env_eval": real_eval,
        "world_model_eval": imagined_evals,
        "teacher_forced_one_step": teacher_forced,
        "open_loop_horizons": horizons,
    }

    print(json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
