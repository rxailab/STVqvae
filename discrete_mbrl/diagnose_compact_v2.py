#!/usr/bin/env python3
"""
Diagnostic for vqvae_wm_compact_v2:
  - Loads the compact_v2 VQVAE encoder (filter_size=5, 25 tokens) and discrete
    transition model directly from their checkpoint files.
  - Draws trajectories from the policy_20k replay buffer (real strong-policy
    data) instead of needing a combined model-free checkpoint.
  - Reports:
      1. Teacher-forced vs open-loop state prediction error at horizons 1/3/5/10
      2. Reward head hallucination: mean predicted reward vs mean actual reward
         on all transitions, and specifically on zero-reward transitions.
"""

import os
import sys
import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model, construct_trans_model


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR    = "/home/xiar3/experiments/STVqvae/wm_runs/vqvae_wm_compact_v2"
ENCODER_HASH = "0e0e6e70075468d257209127edcf6879"
TRANS_HASH   = "da495e284f45100f0889e70f375d54c1"
BUFFER_PATH  = "/home/xiar3/experiments/STVqvae/discrete_mbrl/data/" \
               "MiniGrid-LavaCrossingS9N1-v0_replay_buffer_policy_20k.hdf5"
ENV_NAME     = "MiniGrid-LavaCrossingS9N1-v0"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS     = [1, 3, 5, 10]
N_TRAJ       = 200   # trajectory segments to sample from the buffer


# ---------------------------------------------------------------------------
# Load encoder
# ---------------------------------------------------------------------------
def load_encoder():
    ckpt_path = os.path.join(MODEL_DIR, "models", ENV_NAME,
                             f"model_{ENCODER_HASH}.pt")
    # Checkpoint is raw state_dict (no embedded args) — hardcode compact_v2 settings
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    args = SimpleNamespace(
        env_name        = ENV_NAME,
        device          = DEVICE,
        model_dir       = MODEL_DIR,
        ae_model_type   = "vqvae",
        ae_model_version= "2",
        ae_model_hash   = ENCODER_HASH,
        filter_size     = 5,
        codebook_size   = 256,
        embedding_dim   = 128,
        latent_dim      = None,
        commitment_cost = 0.25,
        ema_decay       = 0.99,
        entropy_penalty_coef = 0.0,
        code_dropout_rate    = 0.0,
        mae_mask_ratio       = 0.0,
        mae_patch_size       = 4,
        mae_loss_coef        = 1.0,
        ctx_aux_coef         = 1.0,
        ctx_channels         = 64,
        ctx_cond_type        = "concat",
        e2e_loss             = False,
        ae_grad_clip         = 0.0,
        env_max_steps        = None,
        wandb                = False,
        comet_ml             = False,
        obs_resize           = None,
        obs_resize_mode      = "bilinear",
        no_obs_resize        = False,
        trans_model_type     = "discrete",
        trans_model_version  = "1",
        trans_model_hash     = TRANS_HASH,
        trans_hidden         = 512,
        trans_depth          = 5,
        stochastic           = "simple",
        n_train_unroll       = 8,
        vq_trans_loss_type   = "mse",
        vq_trans_1d_conv     = False,
        vq_trans_state_snap  = False,
        use_soft_embeds      = False,
        load                 = False,
        save                 = False,
        cache                = True,
        learning_rate        = 3e-4,
        trans_learning_rate  = 3e-4,
        log_freq             = -1,
        repr_sparsity        = 0,
        sparsity_type        = "random",
        fta_tiles            = 20,
        fta_bound_low        = -2,
        fta_bound_high       = 2,
        fta_eta              = 0.2,
        log_norms            = False,
    )

    env = make_env(ENV_NAME)
    reset_result = env.reset()
    sample_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([sample_obs])
    env.close()

    encoder = construct_ae_model(sample_obs.shape[1:], args, load=False)[0]
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Encoder loaded: {args.ae_model_type} filter_size={args.filter_size} "
          f"n_latents={encoder.n_latent_embeds} codebook={args.codebook_size}")
    return encoder, args


# ---------------------------------------------------------------------------
# Load transition model
# ---------------------------------------------------------------------------
def load_transition(encoder, args):
    env = make_env(ENV_NAME)
    t_args = SimpleNamespace(**vars(args))
    t_args.trans_model_hash    = TRANS_HASH
    t_args.trans_model_type    = "discrete"
    t_args.trans_model_version = "1"
    t_args.e2e_loss            = False
    # Build architecture without loading (hash recomputed from args won't match)
    trans_model = construct_trans_model(encoder, t_args, env.action_space, load=False)[0]
    env.close()
    # Load weights directly by the known file path
    ckpt_path = os.path.join(MODEL_DIR, "models", ENV_NAME,
                             f"model_{TRANS_HASH}.pt")
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # checkpoint may be raw state_dict or wrapped dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    trans_model.load_state_dict(state_dict)
    trans_model = trans_model.to(DEVICE).eval()
    for p in trans_model.parameters():
        p.requires_grad_(False)
    print(f"Transition model loaded: hash={TRANS_HASH}")
    return trans_model


# ---------------------------------------------------------------------------
# Load trajectories from replay buffer (episode-reconstructed)
# ---------------------------------------------------------------------------
def load_buffer_episodes():
    """
    Reconstruct episode sequences from the HDF5 buffer.
    Each row is (obs, action, next_obs, reward, done).
    We stitch consecutive rows into episodes using the done flag.
    """
    with h5py.File(BUFFER_PATH, "r") as f:
        obs    = f["obs"][:]           # (N, C, H, W)
        actions = f["action"][:]       # (N,)
        rewards = f["reward"][:]       # (N,)
        dones   = f["done"][:]         # (N,)
        next_obs = f["next_obs"][:]    # (N, C, H, W)

    print(f"Buffer loaded: {len(obs)} transitions, "
          f"positive reward rate={rewards.mean():.4f}")

    # Stitch into episodes
    episodes = []
    ep_obs, ep_acts, ep_rews = [], [], []
    for i in range(len(obs)):
        ep_obs.append(obs[i])
        ep_acts.append(int(actions[i]))
        ep_rews.append(float(rewards[i]))
        if dones[i] or i == len(obs) - 1:
            if len(ep_acts) >= 2:
                # Add final next_obs
                full_obs = ep_obs + [next_obs[i]]
                episodes.append({
                    "obs":     np.stack(full_obs).astype(np.float32),
                    "actions": np.array(ep_acts, dtype=np.int64),
                    "rewards": np.array(ep_rews, dtype=np.float32),
                })
            ep_obs, ep_acts, ep_rews = [], [], []

    print(f"Reconstructed {len(episodes)} episodes, "
          f"mean length={np.mean([len(e['actions']) for e in episodes]):.1f}")
    return episodes


# ---------------------------------------------------------------------------
# Encode episodes into latent token sequences
# ---------------------------------------------------------------------------
def encode_episodes(encoder, episodes):
    encoded = []
    for ep in tqdm(episodes, desc="Encoding"):
        obs_t = torch.from_numpy(ep["obs"]).float().to(DEVICE)
        # Normalise to [0,1] if needed
        if obs_t.max() > 1.5:
            obs_t = obs_t / 255.0
        with torch.no_grad():
            z = encoder.encode(obs_t).long().cpu()  # (T+1, n_latents)
        encoded.append({
            "z":       z,
            "actions": torch.from_numpy(ep["actions"]).long(),
            "rewards": torch.from_numpy(ep["rewards"]).float(),
            "len":     len(ep["actions"]),
        })
    return encoded


# ---------------------------------------------------------------------------
# State prediction metrics
# ---------------------------------------------------------------------------
def discrete_state_metrics(logits, targets):
    # logits: (B, codebook, n_latents)  — 3-D output from DiscreteTransitionModel
    #      OR (B, n_latents * codebook) — flattened alternative
    # targets: (B, n_latents)
    n_latents = targets.shape[-1]
    if logits.dim() == 3:
        # already (B, codebook, n_latents) — use directly
        logits_r = logits                                        # (B, C, L)
    else:
        codebook  = logits.shape[-1] // n_latents
        logits_r  = logits.view(-1, codebook, n_latents)        # (B, C, L)
    targets_r = targets.view(-1, n_latents)                      # (B, L)
    ce = F.cross_entropy(logits_r, targets_r, reduction="none")  # (B, L)
    ce = ce.sum(dim=-1)                                           # (B,)
    preds = logits_r.argmax(dim=1)                               # (B, L)
    acc   = (preds == targets_r).float().mean(dim=-1)            # (B,)
    return ce.mean().item(), acc.mean().item()


# ---------------------------------------------------------------------------
# Horizon evaluation
# ---------------------------------------------------------------------------
def evaluate_horizon(trans_model, encoded_eps, horizon):
    starts, act_seqs, targets = [], [], []
    tf_states, tf_acts, tf_targets = [], [], []
    for ep in encoded_eps:
        T = ep["len"]
        if T < horizon:
            continue
        z = ep["z"]
        a = ep["actions"]
        for t in range(T - horizon + 1):
            starts.append(z[t])
            act_seqs.append(a[t:t + horizon])
            targets.append(z[t + horizon])
            tf_states.append(z[t:t + horizon])
            tf_acts.append(a[t:t + horizon])
            tf_targets.append(z[t + 1:t + horizon + 1])

    if not starts:
        return None

    starts    = torch.stack(starts).to(DEVICE)
    act_seqs  = torch.stack(act_seqs).to(DEVICE)
    tgts      = torch.stack(targets).to(DEVICE)
    tf_s      = torch.stack(tf_states).to(DEVICE)
    tf_a      = torch.stack(tf_acts).to(DEVICE)
    tf_t      = torch.stack(tf_targets).to(DEVICE)

    with torch.no_grad():
        # Teacher-forced: each step uses real latent as input
        tf_logits, _, _ = trans_model(
            tf_s.reshape(-1, tf_s.shape[-1]),
            tf_a.reshape(-1),
            return_logits=True,
        )
        tf_ce, tf_acc = discrete_state_metrics(tf_logits,
                                               tf_t.reshape(-1, tf_t.shape[-1]))

        # Open-loop: roll forward from start using predicted latents
        curr = starts
        for step in range(act_seqs.shape[1]):
            ol_logits, _, _ = trans_model(curr, act_seqs[:, step], return_logits=True)
            curr = trans_model.logits_to_state(ol_logits)
        ol_ce, ol_acc = discrete_state_metrics(ol_logits, tgts)

    return {
        "segments":              len(starts),
        "teacher_forced_loss":   round(tf_ce, 4),
        "teacher_forced_acc":    round(tf_acc, 4),
        "open_loop_loss":        round(ol_ce, 4),
        "open_loop_acc":         round(ol_acc, 4),
    }


# ---------------------------------------------------------------------------
# Reward hallucination check
# ---------------------------------------------------------------------------
def evaluate_reward_hallucination(trans_model, encoded_eps):
    """
    For every 1-step transition, compare the transition model's predicted
    reward against the actual reward. Key signal: on zero-reward transitions
    (the vast majority), does the model predict near-zero, or does it
    hallucinate positive reward?
    """
    pred_rewards, actual_rewards = [], []

    for ep in encoded_eps:
        T = ep["len"]
        if T < 1:
            continue
        z = ep["z"][:-1].to(DEVICE)   # (T, n_latents) – current states
        a = ep["actions"].to(DEVICE)   # (T,)
        r = ep["rewards"]              # (T,) cpu

        with torch.no_grad():
            _, r_pred, _ = trans_model(z, a, return_logits=False)
        pred_rewards.append(r_pred.squeeze(-1).cpu())
        actual_rewards.append(r)

    pred_r  = torch.cat(pred_rewards).numpy()
    actual_r = torch.cat(actual_rewards).numpy()

    zero_mask = actual_r == 0.0
    goal_mask = actual_r  > 0.0

    return {
        "n_transitions":           int(len(pred_r)),
        "actual_reward_mean":      round(float(actual_r.mean()), 6),
        "predicted_reward_mean":   round(float(pred_r.mean()), 6),
        "zero_reward_pred_mean":   round(float(pred_r[zero_mask].mean()), 6) if zero_mask.any() else None,
        "zero_reward_pred_max":    round(float(pred_r[zero_mask].max()),  6) if zero_mask.any() else None,
        "goal_reward_pred_mean":   round(float(pred_r[goal_mask].mean()), 6) if goal_mask.any() else None,
        "goal_reward_actual_mean": round(float(actual_r[goal_mask].mean()), 6) if goal_mask.any() else None,
        "n_zero_transitions":      int(zero_mask.sum()),
        "n_goal_transitions":      int(goal_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    encoder, args = load_encoder()
    trans_model   = load_transition(encoder, args)
    episodes      = load_buffer_episodes()
    encoded_eps   = encode_episodes(encoder, episodes)

    results = {"state_prediction": {}, "reward_hallucination": {}}

    print("\n=== State Prediction (teacher-forced vs open-loop) ===")
    print(f"{'Horizon':>8}  {'Segments':>9}  {'TF loss':>9}  {'TF acc':>8}  "
          f"{'OL loss':>9}  {'OL acc':>8}")
    for h in HORIZONS:
        r = evaluate_horizon(trans_model, encoded_eps, h)
        if r is None:
            print(f"{h:>8}  {'N/A':>9}")
            continue
        results["state_prediction"][h] = r
        print(f"{h:>8}  {r['segments']:>9}  {r['teacher_forced_loss']:>9.4f}  "
              f"{r['teacher_forced_acc']:>8.4f}  {r['open_loop_loss']:>9.4f}  "
              f"{r['open_loop_acc']:>8.4f}")

    print("\n=== Reward Hallucination Check ===")
    rh = evaluate_reward_hallucination(trans_model, encoded_eps)
    results["reward_hallucination"] = rh
    print(f"  Total transitions       : {rh['n_transitions']}")
    print(f"  Actual reward mean      : {rh['actual_reward_mean']:.6f}")
    print(f"  Predicted reward mean   : {rh['predicted_reward_mean']:.6f}")
    print(f"  Zero-reward transitions : {rh['n_zero_transitions']}")
    print(f"    pred mean on zero-r   : {rh['zero_reward_pred_mean']:.6f}")
    print(f"    pred max  on zero-r   : {rh['zero_reward_pred_max']:.6f}")
    print(f"  Goal transitions        : {rh['n_goal_transitions']}")
    if rh['goal_reward_pred_mean'] is not None:
        print(f"    actual mean on goal   : {rh['goal_reward_actual_mean']:.6f}")
        print(f"    pred   mean on goal   : {rh['goal_reward_pred_mean']:.6f}")

    out_path = "/home/xiar3/experiments/vqvae_wm_compact_v2_diag.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
