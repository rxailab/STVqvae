#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model, construct_trans_model


def strip_compiled_prefix(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_policy_checkpoint(model_path, env_name, device, model_dir):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {}).copy()
    saved["env_name"] = env_name
    saved["device"] = device
    saved["model_dir"] = model_dir
    saved.setdefault("wandb", False)
    saved.setdefault("comet_ml", False)
    saved.setdefault("ae_grad_clip", 0.0)
    saved.setdefault("env_max_steps", None)
    return ckpt, SimpleNamespace(**saved)


def load_encoder(model_path, env_name, device, model_dir):
    ckpt, args = load_policy_checkpoint(model_path, env_name, device, model_dir)
    env = make_env(env_name, max_steps=args.env_max_steps)
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([obs])
    env.close()

    encoder = construct_ae_model(sample_obs.shape[1:], args, load=False)[0]
    encoder.load_state_dict(ckpt["ae_model_state_dict"])
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, args


def load_transition_model(args, encoder, env_name, device, trans_model_hash, trans_model_type, trans_model_version):
    wm_args = SimpleNamespace(**vars(args))
    wm_args.env_name = env_name
    wm_args.device = device
    wm_args.model_dir = getattr(args, "model_dir", ".")
    wm_args.trans_model_hash = trans_model_hash
    wm_args.trans_model_type = trans_model_type
    wm_args.trans_model_version = trans_model_version
    wm_args.e2e_loss = False

    env = make_env(env_name, max_steps=wm_args.env_max_steps)
    trans_model = construct_trans_model(encoder, wm_args, env.action_space, load=True)[0]
    act_dim = env.action_space.n
    env.close()

    trans_model = trans_model.to(device).eval()
    for p in trans_model.parameters():
        p.requires_grad = False
    return trans_model, act_dim


def encode_latent(encoder, obs, device):
    obs_tensor = preprocess_obs([obs]).to(device)
    with torch.no_grad():
        return encoder.encode(obs_tensor).long()


def build_action_sequences(act_dim, horizon, device):
    seqs = list(itertools.product(range(act_dim), repeat=horizon))
    return torch.tensor(seqs, dtype=torch.long, device=device)


def score_action_sequences(trans_model, start_latent, action_seqs, gamma_discount):
    n_seq, horizon = action_seqs.shape
    curr = start_latent.repeat(n_seq, 1)
    cumulative = torch.ones(n_seq, device=action_seqs.device)
    scores = torch.zeros(n_seq, device=action_seqs.device)

    with torch.no_grad():
        for t in range(horizon):
            next_latent, reward, gamma = trans_model(curr, action_seqs[:, t])
            reward = reward.squeeze(-1)
            gamma = gamma.squeeze(-1)
            scores = scores + cumulative * reward
            cumulative = cumulative * gamma * gamma_discount
            curr = next_latent.long()
    return scores


def evaluate_mpc(env, encoder, trans_model, act_dim, device, n_episodes, max_steps,
                 horizon, gamma_discount):
    action_seqs = build_action_sequences(act_dim, horizon, device)
    results = defaultdict(list)

    for _ in tqdm(range(n_episodes), desc="Evaluating latent MPC"):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        done = False
        total_reward = 0.0
        step = 0
        success = False

        while not done and step < max_steps:
            start_latent = encode_latent(encoder, obs, device)
            scores = score_action_sequences(
                trans_model, start_latent, action_seqs, gamma_discount
            )
            best_seq = action_seqs[scores.argmax()].detach().cpu().numpy()
            action = int(best_seq[0])

            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = step_result

            reward = float(reward)
            total_reward += reward
            step += 1
            if reward > 0:
                success = True

        results["reward"].append(total_reward)
        results["length"].append(step)
        results["success"].append(float(success))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--trans_model_hash", type=str, required=True)
    parser.add_argument("--trans_model_type", type=str, default="discrete")
    parser.add_argument("--trans_model_version", type=str, default="1")
    parser.add_argument("--model_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--gamma_discount", type=float, default=0.99)
    parser.add_argument("--random_seeds", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    encoder, policy_args = load_encoder(args.model_path, args.env_name, device, args.model_dir)
    trans_model, act_dim = load_transition_model(
        policy_args, encoder, args.env_name, device,
        args.trans_model_hash, args.trans_model_type, args.trans_model_version
    )

    env = make_env(args.env_name, max_steps=args.max_steps, random_seeds=args.random_seeds)
    results = evaluate_mpc(
        env, encoder, trans_model, act_dim, device,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        horizon=args.horizon,
        gamma_discount=args.gamma_discount,
    )
    env.close()

    summary = {
        "env_name": args.env_name,
        "model_path": args.model_path,
        "trans_model_hash": args.trans_model_hash,
        "horizon": args.horizon,
        "n_episodes": args.n_episodes,
        "reward_mean": float(np.mean(results["reward"])),
        "reward_std": float(np.std(results["reward"])),
        "length_mean": float(np.mean(results["length"])),
        "length_std": float(np.std(results["length"])),
        "success_rate": float(np.mean(results["success"])),
        "n_success": int(np.sum(results["success"])),
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
