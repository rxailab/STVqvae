#!/usr/bin/env python3
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model, construct_trans_model
from model_free.rl_utils import interpret_layer_sizes
from shared.models import mlp


def strip_compiled_prefix(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def checkpoint_to_args(model_path, env_name, device, model_dir):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {}).copy()
    saved["env_name"] = env_name
    saved["device"] = device
    saved["model_dir"] = model_dir
    saved.setdefault("wandb", False)
    saved.setdefault("comet_ml", False)
    saved.setdefault("ae_grad_clip", 0.0)
    saved.setdefault("env_max_steps", None)
    saved.setdefault("trans_model_hash", None)
    return ckpt, SimpleNamespace(**saved)


def load_encoder_and_policy(model_path, env_name, device, model_dir):
    ckpt, args = checkpoint_to_args(model_path, env_name, device, model_dir)

    env = make_env(env_name, max_steps=args.env_max_steps)
    reset_result = env.reset()
    sample_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([sample_obs])
    act_dim = env.action_space.n
    env.close()

    encoder = construct_ae_model(sample_obs.shape[1:], args, load=False)[0]
    encoder.load_state_dict(ckpt["ae_model_state_dict"])
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    if args.ae_model_type == "vqvae":
        input_dim = args.embedding_dim * encoder.n_latent_embeds
    else:
        input_dim = encoder.latent_dim

    policy_hidden = interpret_layer_sizes(getattr(args, "policy_hidden", [256, 256]))
    vqvae_e2e = (args.ae_model_type == "vqvae" and getattr(args, "e2e_loss", False))
    mlp_kwargs = {
        "activation": getattr(args, "rl_activation", "relu"),
        "discrete_input": (args.ae_model_type == "vqvae") and (not vqvae_e2e),
    }
    if args.ae_model_type == "vqvae" and not vqvae_e2e:
        mlp_kwargs["n_embeds"] = args.codebook_size
        mlp_kwargs["embed_dim"] = args.embedding_dim

    policy = mlp([input_dim] + policy_hidden + [act_dim], **mlp_kwargs)
    policy.load_state_dict(strip_compiled_prefix(ckpt["policy_state_dict"]))
    policy = policy.to(device).eval()

    return encoder, policy, args


def load_transition_model(args, encoder, env_name, device,
                          trans_model_hash, trans_model_type, trans_model_version):
    args = SimpleNamespace(**vars(args))
    args.env_name = env_name
    args.device = device
    args.trans_model_hash = trans_model_hash
    args.trans_model_type = trans_model_type
    args.trans_model_version = trans_model_version
    args.e2e_loss = False
    args.model_dir = getattr(args, "model_dir", ".")
    env = make_env(env_name, max_steps=getattr(args, "env_max_steps", None))
    trans_model = construct_trans_model(encoder, args, env.action_space, load=True)[0]
    env.close()
    trans_model = trans_model.to(device).eval()
    for p in trans_model.parameters():
        p.requires_grad = False
    return trans_model


def encode_policy_state(encoder, obs_tensor, args):
    if args.ae_model_type == "vqvae" and getattr(args, "e2e_loss", False):
        return encoder.encode(obs_tensor, return_quantized=True)
    return encoder.encode(obs_tensor, return_one_hot=True)


def act_from_policy(encoder, policy, obs, device, args, deterministic=True):
    obs_tensor = preprocess_obs([obs]).to(device)
    with torch.no_grad():
        state = encode_policy_state(encoder, obs_tensor, args)
        logits = policy(state)
        if deterministic:
            return logits.argmax(dim=-1).item()
        return torch.distributions.Categorical(logits=logits).sample().item()


def collect_policy_episodes(encoder, policy, args, device, n_episodes, max_steps, deterministic, random_seeds):
    env = make_env(args.env_name, max_steps=max_steps, random_seeds=random_seeds)
    episodes = []
    for _ in tqdm(range(n_episodes), desc="Collecting real-policy episodes"):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs_seq = [obs]
        act_seq = []
        rew_seq = []
        done = False
        step = 0
        while not done and step < max_steps:
            action = act_from_policy(encoder, policy, obs, device, args, deterministic=deterministic)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _ = step_result
            act_seq.append(action)
            rew_seq.append(float(reward))
            obs_seq.append(next_obs)
            obs = next_obs
            step += 1
        episodes.append({
            "obs": obs_seq,
            "actions": np.array(act_seq, dtype=np.int64),
            "rewards": np.array(rew_seq, dtype=np.float32),
        })
    env.close()
    return episodes


def encode_episode_latents(encoder, episodes, device):
    encoded = []
    for episode in tqdm(episodes, desc="Encoding trajectories"):
        obs_tensor = preprocess_obs(episode["obs"]).to(device)
        with torch.no_grad():
            z = encoder.encode(obs_tensor).long().cpu()
        encoded.append({
            "z": z,
            "actions": torch.from_numpy(episode["actions"]).long(),
            "len": len(episode["actions"]),
        })
    return encoded


def build_horizon_batch(encoded_eps, horizon):
    start_states = []
    action_seqs = []
    target_states = []
    teacher_states = []
    teacher_actions = []
    teacher_targets = []

    for ep in encoded_eps:
        T = ep["len"]
        if T < horizon:
            continue
        z = ep["z"]
        actions = ep["actions"]
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
        "action_seqs": torch.stack(action_seqs),
        "target_states": torch.stack(target_states),
        "teacher_states": torch.stack(teacher_states),
        "teacher_actions": torch.stack(teacher_actions),
        "teacher_targets": torch.stack(teacher_targets),
    }


def discrete_state_metrics(logits, targets):
    if logits.ndim == targets.ndim + 1 and logits.shape[-1] != targets.shape[-1]:
        # Transformer models emit [batch, seq, vocab]; cross_entropy expects class dim second.
        logits = logits.permute(0, 2, 1)
    ce = F.cross_entropy(logits, targets, reduction="none")
    ce = ce.view(targets.shape[0], -1).sum(dim=1)
    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().view(targets.shape[0], -1).mean(dim=1)
    return ce, acc


def evaluate_horizon(trans_model, batch, device):
    start_states = batch["start_states"].to(device)
    action_seqs = batch["action_seqs"].to(device)
    target_states = batch["target_states"].to(device)
    teacher_states = batch["teacher_states"].to(device)
    teacher_actions = batch["teacher_actions"].to(device)
    teacher_targets = batch["teacher_targets"].to(device)

    with torch.no_grad():
        tf_logits, _, _ = trans_model(
            teacher_states.reshape(-1, teacher_states.shape[-1]),
            teacher_actions.reshape(-1),
            return_logits=True,
        )
        tf_targets = teacher_targets.reshape(-1, teacher_targets.shape[-1])
        tf_ce, tf_acc = discrete_state_metrics(tf_logits, tf_targets)

        curr = start_states
        final_logits = None
        for step_idx in range(action_seqs.shape[1]):
            final_logits, _, _ = trans_model(curr, action_seqs[:, step_idx], return_logits=True)
            if getattr(trans_model, "model_type", "").lower() == "transformerdec":
                curr = final_logits.argmax(dim=-1)
            else:
                curr = trans_model.logits_to_state(final_logits)
        ol_ce, ol_acc = discrete_state_metrics(final_logits, target_states)

    return {
        "segments": int(start_states.shape[0]),
        "teacher_forced_state_loss": float(tf_ce.mean().item()),
        "teacher_forced_state_acc": float(tf_acc.mean().item()),
        "open_loop_state_loss": float(ol_ce.mean().item()),
        "open_loop_state_acc": float(ol_acc.mean().item()),
    }


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
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--stochastic_policy", action="store_true")
    parser.add_argument("--random_seeds", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    encoder, policy, policy_args = load_encoder_and_policy(
        args.model_path, args.env_name, device, args.model_dir
    )
    trans_model = load_transition_model(
        policy_args, encoder, args.env_name, device,
        args.trans_model_hash, args.trans_model_type, args.trans_model_version
    )

    episodes = collect_policy_episodes(
        encoder, policy, policy_args, device,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        deterministic=not args.stochastic_policy,
        random_seeds=args.random_seeds,
    )
    encoded_eps = encode_episode_latents(encoder, episodes, device)

    results = {
        "env_name": args.env_name,
        "model_path": args.model_path,
        "trans_model_hash": args.trans_model_hash,
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "policy_deterministic": not args.stochastic_policy,
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
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
