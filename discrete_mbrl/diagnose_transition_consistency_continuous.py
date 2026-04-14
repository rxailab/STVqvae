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

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model, construct_trans_model


class GymnasiumCompatWrapper(gymnasium.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {"render_modes": []})

    def reset(self, seed=None, options=None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        result = self.env.reset()
        if isinstance(result, tuple):
            return result
        return result, {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            return result
        obs, reward, done, info = result
        return obs, reward, bool(done), False, info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class ObsEncoderWrapperGymnasium(gymnasium.ObservationWrapper):
    def __init__(self, env, encoder):
        if not isinstance(env, gymnasium.Env):
            env = GymnasiumCompatWrapper(env)
        super().__init__(env)
        self.encoder = encoder
        self.device = next(self.encoder.parameters()).device

        test_input = np.ones(env.observation_space.shape)
        test_input = preprocess_obs([test_input])
        with torch.no_grad():
            obs_shape = self.encoder.encode(test_input.to(self.device)).shape[1:]

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        obs_tensor = preprocess_obs([obs])
        with torch.no_grad():
            encoded = self.encoder.encode(obs_tensor.to(self.device))[0]
        return encoded.cpu().numpy()


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


def encode_episode_latents(encoder, obs_seq, device):
    obs_tensor = preprocess_obs(obs_seq).to(device)
    with torch.no_grad():
        return encoder.encode(obs_tensor)


def collect_policy_episodes(encoder, policy, env_name, device, n_episodes, max_steps):
    encoded_env = make_encoded_env(env_name, encoder, max_steps)
    real_env = make_env(env_name, max_steps=max_steps)
    episodes = []

    for _ in tqdm(range(n_episodes), desc="Collecting real-policy episodes"):
        obs = encoded_env.reset()
        reset_result = real_env.reset()
        real_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs_seq = [real_obs]
        action_seq = []
        reward_seq = []

        done = False
        while not done and len(action_seq) < max_steps:
            action, _ = policy.predict(obs, deterministic=True)
            action_int = int(np.asarray(action).reshape(-1)[0])

            obs, reward, done_arr, info = encoded_env.step(action)
            step_result = real_env.step(action_int)
            if len(step_result) == 5:
                next_real_obs, real_reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                next_real_obs, real_reward, done, _ = step_result

            action_seq.append(action_int)
            reward_seq.append(float(real_reward))
            obs_seq.append(next_real_obs)
            real_obs = next_real_obs

            if isinstance(done_arr, np.ndarray):
                done = bool(done_arr.reshape(-1)[0]) or done

        episodes.append({
            "obs": obs_seq,
            "actions": np.asarray(action_seq, dtype=np.int64),
            "rewards": np.asarray(reward_seq, dtype=np.float32),
        })

    encoded_env.close()
    real_env.close()
    return episodes


def build_horizon_batch(encoded_eps, horizon):
    start_states = []
    action_seqs = []
    target_states = []
    teacher_states = []
    teacher_actions = []
    teacher_targets = []

    for episode in encoded_eps:
        z = episode["z"]
        actions = episode["actions"]
        T = len(actions)
        if T < horizon:
            continue
        for t in range(T - horizon + 1):
            start_states.append(z[t])
            action_seqs.append(actions[t:t + horizon])
            target_states.append(z[t + horizon])
            teacher_states.append(z[t:t + horizon])
            teacher_actions.append(actions[t:t + horizon])
            teacher_targets.append(z[t + 1:t + horizon + 1])

    if not start_states:
        return None

    return {
        "start_states": torch.stack(start_states),
        "action_seqs": torch.from_numpy(np.stack(action_seqs)).long(),
        "target_states": torch.stack(target_states),
        "teacher_states": torch.stack(teacher_states),
        "teacher_actions": torch.from_numpy(np.stack(teacher_actions)).long(),
        "teacher_targets": torch.stack(teacher_targets),
    }


def mse_per_sample(pred, target):
    return torch.pow(pred - target, 2).view(pred.shape[0], -1).mean(dim=1)


def evaluate_horizon(trans_model, batch, device):
    start_states = batch["start_states"].to(device)
    action_seqs = batch["action_seqs"].to(device)
    target_states = batch["target_states"].to(device)
    teacher_states = batch["teacher_states"].to(device)
    teacher_actions = batch["teacher_actions"].to(device)
    teacher_targets = batch["teacher_targets"].to(device)

    with torch.no_grad():
        tf_pred, _, _ = trans_model(
            teacher_states.reshape(-1, teacher_states.shape[-1]),
            teacher_actions.reshape(-1),
        )
        tf_target = teacher_targets.reshape(-1, teacher_targets.shape[-1])
        tf_mse = mse_per_sample(tf_pred, tf_target)

        curr = start_states
        for step_idx in range(action_seqs.shape[1]):
            curr, _, _ = trans_model(curr, action_seqs[:, step_idx])
        ol_mse = mse_per_sample(curr, target_states)

    return {
        "segments": int(start_states.shape[0]),
        "teacher_forced_state_mse": float(tf_mse.mean().item()),
        "open_loop_state_mse": float(ol_mse.mean().item()),
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 5, 8, 12])
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    args = build_args(args)
    args.device = device

    encoder = load_encoder(args, args.env_name, device)
    trans_model = load_transition_model(args, encoder, args.env_name, device)
    policy = PPO.load(args.policy_path, device=device)

    episodes = collect_policy_episodes(
        encoder, policy, args.env_name, device, args.n_episodes, args.max_steps
    )
    encoded_eps = []
    for episode in tqdm(episodes, desc="Encoding trajectories"):
        encoded_eps.append({
            "z": encode_episode_latents(encoder, episode["obs"], device).cpu(),
            "actions": episode["actions"],
        })

    results = {
        "env_name": args.env_name,
        "policy_path": args.policy_path,
        "model_dir": args.model_dir,
        "ae_model_hash": args.ae_model_hash,
        "trans_model_hash": args.trans_model_hash,
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "horizons": {},
    }

    for horizon in args.horizons:
        batch = build_horizon_batch(encoded_eps, horizon)
        if batch is None:
            results["horizons"][str(horizon)] = {"segments": 0}
            continue
        results["horizons"][str(horizon)] = evaluate_horizon(trans_model, batch, device)

    print(json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
