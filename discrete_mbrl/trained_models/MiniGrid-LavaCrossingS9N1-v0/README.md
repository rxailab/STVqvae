# Trained Models — MiniGrid-LavaCrossingS9N1-v0

Environment: MiniGrid-LavaCrossingS9N1-v0
- 9x9 grid, 1 lava river with 1 gap
- Sparse reward: 1 - 0.9*(steps/max_steps) on success, 0 otherwise
- Max 500 steps per episode
- Action space: 7 discrete actions (turn left, turn right, move forward, pickup, drop, toggle, done)
- Observation: 56x56 RGB image (preprocessed)

---

## Results Summary

| Model | File | Best Avg Reward | Overall Avg | Success at End |
|-------|------|----------------|-------------|----------------|
| PPO + CNN (baseline) | `ppo_cnn_baseline.sb3` | ~0.99 | ~0.99 | ~99% |
| VAE + PPO (e2e) | `vae_e2e_best.pt` | 0.998 | 0.523 | ~99% |
| VQ-VAE + PPO (e2e, gradient fixed) | `vqvae_e2e_best.pt` | 0.686 | 0.075 | Collapsed |

Reward values reflect: 1 - 0.9*(steps/500). A reward of ~0.97 means the episode
was solved in roughly 15 steps.

---

## Model 1: PPO + CNN Baseline

**File:** `ppo_cnn_baseline.sb3`
**Format:** Stable-Baselines3 ActorCriticPolicy (`.sb3`)
**Architecture:**
- Feature extractor: NatureCNN (3 conv layers: 32/64/64 filters, kernel 8/4/3, stride 4/2/1)
- Policy head: MLP 512 -> 7 (action logits)
- Critic head: MLP 512 -> 1 (value)

**Training setup:**
- Framework: Stable-Baselines3 PPO
- Policy type: CnnPolicy
- Parallel envs: 8 (DummyVecEnv)
- Total steps: 300,000
- Eval freq: 30,000 steps, 20 eval episodes

**Key hyperparameters (SB3 defaults):**
- Learning rate: 3e-4
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.0
- vf_coef: 0.5

**Training script:** `discrete_mbrl/eval_policies/train_policy.py`

**Run command:**
```bash
cd discrete_mbrl/eval_policies
python train_policy.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --goal_type goal \
  --n_envs 8 \
  --train_steps 300000
```

**Loading:**
```python
from stable_baselines3.common.policies import ActorCriticPolicy
model = ActorCriticPolicy.load("ppo_cnn_baseline.sb3")
action, _ = model.predict(obs, deterministic=True)
```

---

## Model 2: VAE + PPO (End-to-End)

**File:** `vae_e2e_best.pt`
**Format:** Custom PyTorch checkpoint (dict)
**Architecture:**
- Encoder: Convolutional VAE, filter_size=9, latent_dim=256
- Policy: MLP 256 -> 256 -> 256 -> 7
- Critic: MLP 256 -> 256 -> 256 -> 1

**How it works:**
The VAE encoder maps the 56x56 RGB observation to a 256-dimensional continuous
latent vector via the reparameterization trick. The policy MLP takes this vector
directly as input. With e2e_loss, PPO gradients flow back through the encoder,
adapting the representation to be useful for the policy.

**Training setup:**
- Framework: Custom PPO implementation (ppo.py)
- Total steps: 3,000,000
- Batch size (rollout): 512 steps
- PPO epochs per batch: 20
- PPO minibatch size: 64
- Entropy coef: 0.01
- Learning rate: 1e-4 (shared encoder + policy)
- e2e_loss: True (encoder trained jointly with policy)

**Training script:** `discrete_mbrl/model_free/train.py`

**Run command:**
```bash
cd discrete_mbrl/model_free
PYTHONPATH="/path/to/Vqvae2Path" python train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vae \
  --filter_size 9 \
  --latent_dim 256 \
  --mf_steps 3000000 \
  --batch_size 512 \
  --ppo_iters 20 \
  --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 \
  --e2e_loss \
  --learning_rate 1e-4 \
  --device cuda \
  --save
```

**Loading:**
```python
import torch
from argparse import Namespace

ckpt = torch.load("vae_e2e_best.pt", map_location="cpu", weights_only=False)
# ckpt keys: step, avg_reward, ae_model_state_dict, policy_state_dict,
#            critic_state_dict, optimizer_state_dict, args, model_info
print(ckpt["avg_reward"])   # best rolling average reward
print(ckpt["args"])         # all training hyperparameters
```

**Visualize rollout:**
```bash
cd discrete_mbrl/model_free
PYTHONPATH="/path/to/Vqvae2Path" python visualize_rollout.py \
  --model_path ../trained_models/MiniGrid-LavaCrossingS9N1-v0/vae_e2e_best.pt \
  --n_episodes 5
```

---

## Model 3: VQ-VAE + PPO (End-to-End, Gradient Fixed)

**File:** `vqvae_e2e_best.pt`
**Format:** Custom PyTorch checkpoint (dict)
**Architecture:**
- Encoder: VQ-VAE v2, filter_size=9, codebook_size=64, embedding_dim=64
  - Produces 81 spatial latents (9x9 grid), each assigned to one of 64 codebook vectors
  - Total policy input: 64 * 81 = 5,184 dimensions (flat quantized embeddings)
- Policy: MLP 5184 -> 256 -> 256 -> 7
- Critic: MLP 5184 -> 256 -> 256 -> 1

**How it works:**
The VQ-VAE maps each spatial position in the feature map to the nearest vector
in a learned codebook of 64 entries. The policy receives the flat concatenation
of all 81 quantized embedding vectors as a continuous float tensor. Gradients
flow back to the encoder via the straight-through estimator.

**Key bugs fixed during development:**
1. `--commitment-cost` arg was defined but never passed to the model (was hardcoded 0.25)
2. EMA decay hardcoded at 0.99; exposed as `--ema-decay` arg
3. Separate encoder LR (`--encoder_lr`) added so encoder updates 10x slower than policy
4. **Critical fix:** e2e gradient path was broken — `ArgmaxLayer` blocked all gradients.
   Fixed by using `return_quantized=True` (continuous quantized embeddings) instead
   of `return_one_hot=True` (discrete indices via non-differentiable ArgmaxLayer)

**Training setup:**
- Framework: Custom PPO implementation (ppo.py)
- Total steps: 3,000,000
- Batch size (rollout): 512 steps
- PPO epochs per batch: 20
- PPO minibatch size: 64
- Entropy coef: 0.01
- Policy/critic learning rate: 1e-4
- Encoder learning rate: 1e-5 (10x lower — prevents codebook instability)
- Commitment cost: 0.5
- EMA decay: 0.9 (faster codebook tracking under RL-driven encoder shifts)
- e2e_loss: True
- Pre-trained VQ-VAE encoder used as initialization (trained on 200k random transitions)

**Training script:** `discrete_mbrl/model_free/train.py`

**Run command:**
```bash
cd discrete_mbrl/model_free
PYTHONPATH="/path/to/Vqvae2Path" python train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae \
  --ae_model_version 2 \
  --codebook_size 64 \
  --embedding_dim 64 \
  --filter_size 9 \
  --mf_steps 3000000 \
  --batch_size 512 \
  --ppo_iters 20 \
  --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 \
  --e2e_loss \
  --learning_rate 1e-4 \
  --encoder_lr 1e-5 \
  --commitment-cost 0.5 \
  --ema-decay 0.9 \
  --model_dir .. \
  --device cuda \
  --save
```

**Loading:**
```python
import torch
ckpt = torch.load("vqvae_e2e_best.pt", map_location="cpu", weights_only=False)
print(ckpt["avg_reward"])   # 0.686
print(ckpt["args"])         # all training hyperparameters
```

**Visualize rollout:**
```bash
cd discrete_mbrl/model_free
PYTHONPATH="/path/to/Vqvae2Path" python visualize_rollout.py \
  --model_path ../trained_models/MiniGrid-LavaCrossingS9N1-v0/vqvae_e2e_best.pt \
  --n_episodes 5
```

---

## Notes on VQ-VAE vs VAE Performance Gap

The VAE reaches ~99% while the best VQ-VAE run reached ~69% (then collapsed).

**Root cause:** VQ-VAE's discrete codebook creates a non-stationary policy input
distribution. As the encoder shifts due to RL gradients, observations near codebook
boundaries get reassigned to different codes — the policy sees a completely different
input for the same observation. The VAE has no such boundary and produces smooth,
stable latents throughout training.

**Mitigations applied (partially effective):**
- Lower encoder LR (1e-5) slows encoder drift
- Faster EMA decay (0.9) makes codebook track encoder changes more quickly
- Separate optimizer param groups for encoder vs policy/critic

**What would likely close the gap further:**
- More training steps (5M+) — the reward was still climbing at 69%
- `--ae_recon_loss` — reconstruction anchor prevents encoder from drifting
- Smaller codebook (e.g., 16 or 32) — fewer boundaries, more stable assignments
