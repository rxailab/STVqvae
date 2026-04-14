# Experiment Log — Vqvae2Path

**Environment:** MiniGrid-LavaCrossingS9N1-v0
**Model:** VQVAE v2 + PPO (end-to-end)
**Codebase:** `/mmfs1/storage/users/xiar3/exp/Vqvae2Path/discrete_mbrl/`

---

## Shared Architecture

### VQVAE (version 2) — `make_ae_v2` in `model_construction.py`

**Encoder** — 3-layer strided CNN:
```
Conv2d(C_in → 64,  kernel=8, stride=2, padding=1) + ReLU
Conv2d(64  → 128,  kernel=6, stride=2, padding=0) + ReLU
Conv2d(128 →  64,  kernel=4, stride=2, padding=0) + ReLU
AdaptiveAvgPool2d(9×9)   [filter_size=9]
```
Output: 81 spatial positions × 64-dim embedding = **5184-dim quantized state**

**VQ layer:** EMA-based, 64 codebook entries, commitment_cost=0.25, ema_decay=0.99

**Decoder** — mirrored ConvTranspose2d layers + ResidualBlock + AdaptiveAvgPool2d to restore input resolution

**Policy / Critic:** MLP [5184 → 256 → 256 → out], activation=ReLU

---

## Shared Hyperparameters (all experiments)

| Parameter | Value |
|---|---|
| `mf_steps` | 5,000,000 |
| `batch_size` | 256 (rollout collection) |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `ppo_clip` | 0.2 |
| `ppo_value_coef` | 0.5 |
| `ppo_entropy_coef` | 0.01 |
| `e2e_loss` | True (PPO gradients flow through VQVAE encoder) |
| `ae_recon_loss` | False (except e2erecon) |
| `use_amp` | True (mixed precision) |
| `accumulation_steps` | 32 |
| `codebook_size` | 64 |
| `embedding_dim` | 64 |
| `filter_size` | 9 |
| `policy_hidden` | [256, 256] |
| `critic_hidden` | [256, 256] |

---

## Experiments

### 1. e2ebaseline

**Job ID:** 20279173
**Status:** Finished
**Script:** `e2e.com`
**Node:** gpu04

**Key settings (differences from e2estable):**

| Parameter | Value |
|---|---|
| `encoder_lr` | same as policy LR (3e-4, no separation) |
| `ppo_iters` | 20 |
| `ppo_batch_size` | 32 |
| `ppo_gae_lambda` | 0 (no GAE) |
| `ppo_norm_advantages` | False |
| `ppo_max_grad_norm` | disabled (0) |
| `ortho_init` | False |
| `entropy_penalty_coef` | 0 (none) |

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.3967** |
| Overall avg reward | 0.0273 |
| Final 10-ep avg | 0.0000 |
| Best reward progression | 0.095 → 0.199 → 0.297 → 0.397 |

**Notes:** Policy collapsed by end of training (final avg = 0). Best model reached ~40% success rate mid-training but did not sustain it.

---

### 2. e2estable

**Job ID:** 20289278
**Status:** Finished
**Script:** `e2e_stable.com`
**Node:** gpu04
**Rollouts:** `./rollouts/e2estable/` (10 GIFs)

**Key settings (changes vs e2ebaseline):**

| Parameter | Value | Rationale |
|---|---|---|
| `encoder_lr` | 3e-5 | 10× lower LR for VQVAE to prevent representation drift |
| `ppo_iters` | 10 | Fewer epochs per batch to reduce on-policy staleness |
| `ppo_batch_size` | 64 | Larger mini-batches |
| `ppo_gae_lambda` | 0.95 | GAE for better advantage estimates |
| `ppo_norm_advantages` | True | Normalize advantages per mini-batch |
| `ppo_max_grad_norm` | 0.5 | Gradient clipping for stability |
| `ortho_init` | True | Orthogonal weight init for policy/critic |
| `entropy_penalty_coef` | 0.05 | Soft entropy penalty to prevent codebook collapse |

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.8970** |
| Overall avg reward | 0.1033 |
| Final 10-ep avg | 0.0000 |
| Best checkpoint step | 4,175,683 |
| Best reward progression | 0.099 → 0.198 → 0.296 → 0.396 → 0.493 → 0.581 → 0.695 → 0.794 → 0.897 |

**Rollout evaluation (deterministic, best checkpoint):**

| Episode | Outcome | Reward | Steps |
|---|---|---|---|
| 1 | SUCCESS | 0.97 | 15 |
| 2 | FAILED | 0.00 | 500 |
| 3 | SUCCESS | 0.97 | 15 |
| 4 | FAILED | 0.00 | 500 |
| 5 | FAILED | 0.00 | 500 |
| 6 | SUCCESS | 0.97 | 14 |
| 7 | FAILED | 0.00 | 500 |
| 8 | FAILED | 0.00 | 500 |
| 9 | FAILED | 0.00 | 14 |
| 10 | FAILED | 0.00 | 500 |

**Eval success rate: 3/10 (30%)**

**Notes:** Significant improvement over e2ebaseline (0.40 → 0.90 peak). The stability improvements (separate encoder LR, GAE, grad clipping, ortho init) all contributed. Policy again collapsed by end of training — best checkpoint is mid-training. The 3/10 eval success on deterministic rollouts is lower than training peak, likely due to layout variability and policy brittleness on unseen seeds.

---

### 3. e2erecon

**Job ID:** 20289738
**Status:** Done
**Script:** `e2e_recon.com`
**Node:** gpu05

**Key settings (changes vs e2ebaseline):**

| Parameter | Value | Rationale |
|---|---|---|
| `ae_recon_loss` | True | Joint reconstruction loss alongside e2e PPO |
| `ae_er_train` | True | Train AE from replay buffer (diverse samples) |
| `encoder_lr` | 3e-5 | Slower encoder updates to balance two loss signals |
| `ppo_gae_lambda` | 0.95 | GAE |
| `ppo_norm_advantages` | True | Normalize advantages |
| `ppo_max_grad_norm` | 0.5 | Gradient clipping |
| `ppo_iters` | 20 | (same as baseline) |
| `ppo_batch_size` | 32 | (same as baseline) |
| `ortho_init` | False | (not added) |
| `entropy_penalty_coef` | 0 | (not added) |

**Hypothesis:** Anchoring the VQVAE with a reconstruction objective prevents the encoder from drifting purely toward RL-useful features and losing spatial structure.

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.2827** |
| Overall avg reward | 0.0095 |
| Final 10-ep avg | 0.0000 |

**Notes:** Worst result so far. The reconstruction loss actively competed with the RL objective, slowing learning significantly. The VQVAE debug logs show stable MSE (~0.002) throughout, suggesting the encoder optimized for reconstruction at the expense of RL-useful representations.

---

### 4. e2ecosine

**Job ID:** 20299312
**Status:** Done
**Script:** `e2ecosine.com`
**Node:** gpu04
**Model files:** `e2ecosine_best_model.pt` / `e2ecosine_final_model.pt`

**Key settings (changes vs e2estable):**

| Parameter | Value | Rationale |
|---|---|---|
| `encoder_lr_cosine` | True | Cosine anneal encoder LR from 3e-5 to 0 over 5M steps |
| (all other settings) | same as e2estable | Isolate effect of cosine decay |

**Hypothesis:** Gradually reducing encoder LR to zero prevents late-training representation drift that causes policy collapse, while allowing the encoder to benefit from RL signal early.

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.6983** |
| Overall avg reward | 0.1490 |
| Final 10-ep avg | 0.1977 |

**Notes:** Best final avg reward of all experiments so far (0.198 vs 0.000 for e2estable). Still collapsed somewhat at the end but much less severely. Cosine decay is the most effective anti-collapse technique tested so far.

---

### 5. e2ephased

**Job ID:** 20299313
**Status:** Done
**Script:** `e2ephased.com`
**Node:** gpu04
**Model files:** `e2ephased_best_model.pt` / `e2ephased_final_model.pt`

**Key settings (changes vs e2estable):**

| Parameter | Value | Rationale |
|---|---|---|
| `encoder_lr_cosine` | True | Cosine anneal encoder LR from 3e-5 to 0 |
| `freeze_encoder_after` | 2,500,000 | Hard-freeze encoder at halfway point |
| (all other settings) | same as e2estable | |

**Hypothesis:** Combining cosine decay with a hard freeze at the midpoint gives the strongest protection against encoder drift. Phase 1 (0-2.5M) allows e2e learning with decaying encoder updates; Phase 2 (2.5M-5M) trains policy only on a fixed representation.

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.5940** |
| Overall avg reward | 0.0809 |
| Final 10-ep avg | 0.0000 |

**Notes:** Encoder froze at step 2,500,096 as intended. Despite the freeze, policy collapsed to 0 by end of training. The hard freeze did not prevent collapse — either the representation was already degraded before the freeze point, or the policy itself became unstable in phase 2 without encoder adaptation.

---

### 6. e2ecosine2

**Job ID:** 20306194
**Status:** Done
**Script:** `e2ecosine2.com`
**Node:** gpu04
**Model files:** `e2ecosine2_best_model.pt` / `e2ecosine2_final_model.pt`

**Key settings (changes vs e2ecosine):**

| Parameter | Value | Rationale |
|---|---|---|
| Bug fix | `LambdaLR` per-group | `CosineAnnealingLR` was decaying policy LR too; now only encoder group is annealed |
| (all other settings) | same as e2ecosine | |

**Hypothesis:** With the bug fixed, the policy LR stays at 3e-4 throughout while only the encoder LR decays 3e-5 -> 0. This should combine e2estable's high peak with e2ecosine's reduced collapse.

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.7982** |
| Overall avg reward | 0.1532 |
| Final 10-ep avg | 0.1881 |

**Notes:** Hypothesis confirmed — fixing the bug raised the peak from 0.698 to 0.798 (approaching e2estable's 0.897), while the final avg (0.188) stayed close to e2ecosine (0.198). The policy LR fix allowed stronger mid-training learning without sacrificing the anti-collapse benefit. Still some late collapse but significantly less than e2estable (0.000).

---

### 7. e2eema

**Status:** Done
**Script:** `discrete_mbrl/model_free/train.py`
**Hardware:** RTX 4090 (local, direct run — no SLURM)
**Model files:** `./models/MiniGrid-LavaCrossingS9N1-v0/e2eema_best_model.pt` / `e2eema_final_model.pt`
**Log:** `/home/xiar3/experiments/e2eema.log`

**Full command:**
```bash
python train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae --ae_model_version 2 \
  --codebook_size 64 --embedding_dim 64 --filter_size 9 \
  --mf_steps 5000000 --batch_size 512 \
  --num_envs 16 \
  --e2e_loss --encoder_lr 3e-5 --encoder_lr_cosine \
  --encoder_ema_tau 0.995 \
  --ppo_iters 10 --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages --ppo_max_grad_norm 0.5 \
  --entropy_penalty_coef 0.05 --ortho_init \
  --run_name e2eema --device cuda --save
```

**Key changes vs e2ecosine2:**

| Parameter | Value | Rationale |
|---|---|---|
| `encoder_ema_tau` | 0.995 | **EMA target encoder** — slow-moving copy (τ=0.995) of online encoder; rollout collection and PPO value bootstrapping use the stable target encoder; PPO gradients still flow through online encoder only |
| `num_envs` | 16 | Vectorized parallel environments (`SyncVectorEnv`) — fixes GPU underutilization (7% → ~30%), throughput 666 → ~1130 env steps/sec |
| `batch_size` | 512 | Doubled from 256 to accommodate 16 envs × 32 steps per update |

**Infrastructure changes (new code):**
- `env_helpers.py`: Added `make_vec_env()` using `gymnasium.vector.SyncVectorEnv`
- `train.py`: Added `_ema_update()` helper, target encoder creation (`deepcopy` of raw ae_model, frozen, eval mode), vectorized rollout path
- `ppo.py`: Added `target_ae` parameter to `PPOTrainer` for stable bootstrap values
- `rl_utils.py`: Added `--num_envs` and `--encoder_ema_tau` CLI arguments

**Hypothesis:** An EMA target encoder (borrowed from DQN target network) gives the policy a stable input distribution during rollouts and value bootstrapping, directly addressing the root cause of policy collapse — encoder drift under PPO gradients. The online encoder still learns via e2e gradients, but the policy never sees its drifting outputs.

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.6625** |
| Overall avg reward | 0.0931 |
| Final 10-ep avg | 0.0000 |

**Notes:** Policy still collapsed to 0 by end of training despite the EMA target encoder. Peak (0.6625) is lower than e2ecosine2 (0.798) and e2estable (0.897). The EMA target encoder did not prevent collapse — the online encoder still drifts under PPO gradients, which eventually corrupts the codebook EMA. The target encoder, being a slow copy of the online encoder, follows the drift with a delay but does not stop it. Also note: the combined effect of `encoder_lr_cosine` + `encoder_ema_tau=0.995` may have been too aggressive — annealing the encoder LR to 0 while also delaying gradient feedback via EMA possibly slowed early learning without improving stability.

---

### 8. e2esnapback

**Status:** Done
**Script:** `discrete_mbrl/model_free/train.py`
**Hardware:** RTX 4090 (local, direct run — no SLURM)
**Model files:** `./models/MiniGrid-LavaCrossingS9N1-v0/e2esnapback_best_model.pt` / `e2esnapback_final_model.pt`
**Log:** `/home/xiar3/experiments/e2esnapback.log`

**Full command:**
```bash
python train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae --ae_model_version 2 \
  --codebook_size 64 --embedding_dim 64 --filter_size 9 \
  --mf_steps 5000000 --batch_size 4096 \
  --num_envs 16 \
  --e2e_loss --encoder_lr 3e-5 \
  --encoder_snapback --snapback_threshold 0.5 --snapback_patience 100 --snapback_min_reward 0.6 \
  --ppo_iters 10 --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages --ppo_max_grad_norm 0.5 \
  --entropy_penalty_coef 0.05 --ortho_init \
  --run_name e2esnapback --device cuda --save
```

**Changes vs e2estable:**

| Parameter | Value | Rationale |
|---|---|---|
| `num_envs` | 16 | Vectorized envs for GPU utilization |
| `batch_size` | **4096** | 16 envs × 256 steps/env = identical per-env trajectory length to e2estable |
| `encoder_snapback` | True | Adaptive encoder snapback (see below) |
| `snapback_threshold` | 0.5 | Trigger when rolling avg drops below 50% of peak |
| `snapback_patience` | 100 | 100 consecutive declining episodes before triggering |
| `snapback_min_reward` | 0.6 | Don't activate until peak ≥ 0.6 (avoids noise-triggered early freeze) |
| `encoder_lr_cosine` | False | Removed — cosine reduced peak (0.798 vs 0.897) |
| `encoder_ema_tau` | 1.0 | Removed — EMA didn't help |

---

#### What went wrong in all previous local runs (root cause analysis)

Every local experiment before this one (e2eema, e2esnapback v1, v2) peaked in the range 0.29–0.66 despite the HPC run e2estable reaching 0.897. Two bugs in the vectorized rollout were responsible.

**Bug 1 — Trajectory too short (`batch_size=512` with `num_envs=16`)**

With 16 environments and `batch_size=512`, each env only rolled out for `512 / 16 = 32 steps` before a PPO update. This environment has episodes up to 500 steps long. Consequences:
- Almost every trajectory segment was truncated mid-episode (never reached a terminal state)
- The GAE bootstrap was called at every 32nd step with a noisy critic value estimate
- Effective credit assignment horizon was tiny — the agent couldn't learn from episode-length consequences
- PPO was seeing almost exclusively mid-episode fragments, never complete episodes

This alone dropped the peak from ~0.9 to 0.47. Fix: `batch_size=4096` → 256 steps per env, matching e2estable's single-env rollout length.

**Bug 2 — Cross-env GAE contamination (interleaved transition ordering)**

The vectorized rollout stored transitions in time-interleaved order:
```
buffer = [t0·e0, t0·e1, ..., t0·e15,  t1·e0, t1·e1, ..., t1·e15,  ...]
position:    0     1          15          16     17          31
```

PPO's GAE computation processes this as a single sequential trajectory — treating position 1 (`t0·e1`, env 1's obs at timestep 0) as the "next state" after position 0 (`t0·e0`, env 0's obs at timestep 0). These are from completely different environments with no temporal relationship. The GAE backward pass:
```python
gae[i] = delta[i] + lambda * gamma[i] * gae[i+1]  # gae[i+1] is from a different env!
```
corrupted every single advantage estimate. The `gamma[i]` mask (which zeros out at episode ends) only partially contained the damage since episodes rarely ended exactly at t0.

Fix: collect all `T` timesteps as `[T, N, ...]` tensors, then:
1. **Bootstrap**: for each env that didn't terminate at step T, compute `V(next_obs)` and add `gamma * V` to that env's last reward, then set its gamma to 0
2. **Permute**: reshape `[T, N, ...] → [N, T, ...] → [N×T, ...]` (env-major order)

Env-major ordering puts env 0's full 256-step trajectory first, env 1's next, etc. The `gamma=0` at each env boundary (from the bootstrap step) ensures GAE's backward pass stops at every boundary and never bleeds from one env into another.

**Impact of each fix:**

| Fix applied | Peak reward |
|---|---|
| Neither (batch_size=512, interleaved) | 0.293 – 0.663 |
| Bug 1 only (batch_size=4096, still interleaved) | 0.473 |
| Both bugs fixed | **0.9988** |

The combination recovered e2estable's peak and then exceeded it.

---

#### What the snapback mechanism does

After each PPO batch, the code checks every completed episode:
1. If the current 10-episode rolling avg is a new best AND ≥ `snapback_min_reward` gate: save a copy of the encoder's `state_dict`
2. If the current avg has been below `best × snapback_threshold` for `snapback_patience` consecutive episodes: restore the encoder to the saved best state_dict and permanently freeze it (set `requires_grad=False` on all encoder params, zero out encoder LR)

The `snapback_min_reward=0.6` gate prevents activation during the noisy early phase when rolling avg swings between 0 and 0.3. Without it, a 10-episode window average of 0.1 dropping to 0.0 (which happens constantly when success rate is 10%) would trigger a premature freeze.

```python
# Simplified logic:
if current_avg > best_avg and current_avg > min_reward:
    best_avg = current_avg
    best_encoder_sd = deepcopy(encoder.state_dict())

if current_avg < best_avg * threshold:
    decline_count += 1
    if decline_count >= patience:
        encoder.load_state_dict(best_encoder_sd)  # restore
        freeze(encoder)                            # permanent freeze
        train_encoder = False
```

---

#### Results

| Metric | Value |
|---|---|
| **Best rolling avg (10-ep window)** | **0.9988** (new all-time best; e2estable was 0.897) |
| Snapback triggered at step | 4,160,736 (~83% into training) |
| Rolling avg at trigger | 0.199 (< 0.5 × 0.9988 = 0.499) |
| **Final 10-ep avg** | **0.3995** |
| vs e2estable final | 0.3995 vs 0.000 — snapback prevented total collapse |

**Phase-by-phase breakdown:**
- **Phase 1 (steps 0–4.16M):** Full e2e training with corrected GAE. Encoder and policy both learn. Peak climbs steadily to 0.9988.
- **Phase 2 (step 4.16M):** Collapse begins. Rolling avg drops from ~0.9988 to 0.199 over ~100 episodes. Snapback fires: encoder restored to peak state_dict and frozen.
- **Phase 3 (steps 4.16M–5M):** Only ~840k steps of policy training with frozen encoder. Policy partially recovers to 0.3995 final.

**What limited Phase 3:** Only 17% of training budget remained after snapback triggered. With a fixed encoder, the policy needs many more steps to re-optimise from scratch on the stable representation.

---

### 9. e2esnapback_10m

**Status:** Done
**Script:** `discrete_mbrl/model_free/train.py`
**Hardware:** RTX 4090 (local, direct run — no SLURM)
**Model files:** `./models/MiniGrid-LavaCrossingS9N1-v0/e2esnapback_10m_best_model.pt`
**Log:** `/tmp/e2esnapback_10m.log`

**Motivation:** e2esnapback hit peak **0.9988** but snapback triggered at step 4.16M (83% through 5M), leaving only ~840k steps of stable policy training post-freeze. The recovery reached only 0.3995 final. By doubling the budget to 10M steps, snapback still triggers around 4M steps but leaves ~6M steps of stable frozen-encoder PPO — giving the policy ~7× more time to consolidate after freeze.

**Only change vs e2esnapback:** `--mf_steps 10000000` (5M → 10M).

**Full command:**
```bash
cd discrete_mbrl/model_free
PYTHONPATH=../.. python train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae --ae_model_version 2 \
  --codebook_size 64 --embedding_dim 64 --filter_size 9 \
  --mf_steps 10000000 --batch_size 4096 \
  --num_envs 16 \
  --e2e_loss --encoder_lr 3e-5 \
  --encoder_snapback --snapback_threshold 0.5 --snapback_patience 100 --snapback_min_reward 0.6 \
  --ppo_iters 10 --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages --ppo_max_grad_norm 0.5 \
  --entropy_penalty_coef 0.05 --ortho_init \
  --run_name e2esnapback_10m --device cuda --save
```

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.9988** |
| Final 10-ep avg | **0.9984** |
| Overall avg reward | **0.9732** |
| Snapback triggered at step | **Never** |

**Analysis:**

Hypothesis confirmed and exceeded. Not only did the final reward jump from 0.3995 → **0.9984**, snapback never triggered at all — the encoder maintained a stable representation for the entire 10M steps without collapsing. Two possible explanations:

1. **Longer training stabilises the encoder:** With 10M steps, the policy becomes highly competent before the encoder has a chance to drift far. A well-trained policy generates more consistent, on-policy gradient signals that keep the encoder in a stable region.
2. **Stochastic timing:** The collapse in the 5M run may have been a near-miss — a small perturbation tipped it over. With the same hyperparameters, a second run might simply not hit that perturbation.

Regardless of mechanism, the result is the new best: **final 0.9984**, **overall 0.9732** — consistently near-optimal throughout training, not just at peak.

**Comparison vs e2esnapback (5M):**

| Metric | e2esnapback (5M) | e2esnapback_10m (10M) |
|---|---|---|
| Peak reward | 0.9988 | **0.9988** |
| Final 10-ep avg | 0.3995 | **0.9984** |
| Overall avg | — | **0.9732** |
| Snapback triggered | Yes (step 4.16M) | **No** |

---

### 10. e2e_ema_tau

**Status:** Done
**Script:** `discrete_mbrl/model_free/train.py`
**Hardware:** RTX 4090 (local, direct run — no SLURM)
**Model files:** `./models/MiniGrid-LavaCrossingS9N1-v0/e2e_ema_tau_best_model.pt`
**Log:** `/tmp/e2e_ema_tau.log`

**Motivation:** In all e2e experiments, the policy observes the *online* encoder output — when the encoder drifts, the policy's input distribution shifts suddenly. With a slow EMA target encoder (`tau=0.995`), the *rollout* encoder is a momentum average of past online weights, changing smoothly rather than jerking. The online encoder still receives full PPO gradients (learns task-relevant features), but the policy always sees a stable, slowly-evolving representation. This decouples learning speed from representation stability.

**Key change vs e2esnapback:** Replace snapback with `--encoder_ema_tau 0.995`. The EMA target is used for rollout collection and bootstrapping; the online encoder updates with e2e gradients.

**Full command:**
```bash
cd discrete_mbrl/model_free
PYTHONPATH=../.. python train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae --ae_model_version 2 \
  --codebook_size 64 --embedding_dim 64 --filter_size 9 \
  --mf_steps 5000000 --batch_size 4096 \
  --num_envs 16 \
  --e2e_loss --encoder_lr 3e-5 \
  --encoder_ema_tau 0.995 \
  --ppo_iters 10 --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages --ppo_max_grad_norm 0.5 \
  --entropy_penalty_coef 0.05 --ortho_init \
  --run_name e2e_ema_tau --device cuda --save
```

**Key differences vs previous experiments:**

| Aspect | e2esnapback | e2e_ema_tau |
|---|---|---|
| Rollout encoder | Online (drifts) | **EMA target (smooth)** |
| Encoder gradients | e2e PPO | **e2e PPO (same)** |
| Anti-collapse mechanism | Snapback (reactive) | **EMA smoothing (proactive)** |
| Post-collapse recovery | ~840k steps (5M run) | **No collapse expected** |

**Hypothesis:** EMA smoothing prevents the sudden input-distribution jumps that destabilise the policy, allowing the encoder to keep learning throughout all 5M steps without a collapse event. Expected final reward higher than e2esnapback's 0.3995, potentially matching or exceeding the 0.9988 peak.

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.3987** |
| Final 10-ep avg | **0.0000** |
| Overall avg reward | **0.0296** |
| Snapback triggered | Never |

**Analysis — why it failed:**

Hypothesis falsified. EMA smoothing (τ=0.995) did not prevent encoder collapse — the final reward dropped to 0.0000 despite no snapback triggering. The peak of 0.3987 is similar to e2esnapback's post-collapse recovery (0.3995), suggesting the EMA target merely delayed the onset of drift rather than preventing it.

Root cause: The *online* encoder receives full PPO gradients and still drifts into a bad representation over 5M steps. The EMA target (used for rollouts) is a lagged copy — it smooths *sudden* jumps but faithfully follows the online encoder's long-term drift. By the time the policy has learned to rely on the target's representation, the target has also drifted far enough to break it.

**Key insight:** EMA decoupling helps over short timescales (episode-to-episode) but not over long training (millions of steps). It is a smoothing mechanism, not a stabilisation mechanism. The only interventions that actually prevent collapse are:
1. **More budget** (e2esnapback_10m) — encoder drift never reached a critical threshold within 10M steps
2. **Reactive freeze** (e2esnapback) — catches collapse after the fact and freezes

**Comparison vs all approaches:**

| Experiment | Peak | Final | Collapse? |
|---|---|---|---|
| e2esnapback (5M) | 0.9988 | 0.3995 | Yes — snapback at step 4.16M |
| **e2esnapback_10m (10M)** | **0.9988** | **0.9984** | **No** |
| e2e_ema_tau (5M, τ=0.995) | 0.3987 | 0.0000 | Yes — gradual drift |
| vqvae_pretrain_ppo (frozen) | 0.1927 | 0.0000 | N/A — no RL signal |

---

### 11. vqvae_pretrain_ppo

**Status:** Done
**Script:** `discrete_mbrl/collect_data.py` → `discrete_mbrl/train_encoder.py` → `discrete_mbrl/model_free/train.py`
**Hardware:** RTX 4090 (local, direct run — no SLURM)
**Model files:** `./models/MiniGrid-LavaCrossingS9N1-v0/vqvae_pretrain_ppo_best_model.pt`
**Log:** `/tmp/vqvae_pretrain_ppo.log`

**Motivation:** In e2esnapback, Phase 3 (policy training on frozen encoder) recovered to 0.3995 in only 840k steps. If we start with a frozen, pre-trained VQVAE encoder from step 0, all 5M steps are pure policy training on a stable representation. The key question is whether a reconstruction-trained VQVAE (no RL signal) provides a good enough representation for PPO to learn from.

**Three-phase pipeline:**

**Phase 1 — Data collection** (`collect_data.py`):
```bash
cd discrete_mbrl
python collect_data.py \
  -e MiniGrid-LavaCrossingS9N1-v0 \
  -n 8 -s 200000 -a random
```
Collects 200k transitions using a random policy → saved to `./data/MiniGrid-LavaCrossingS9N1-v0_replay_buffer.hdf5`.

**Phase 2 — VQVAE pretraining** (`train_encoder.py`):
```bash
python train_encoder.py \
  -e MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae --ae_model_version 2 \
  --codebook_size 64 --embedding_dim 64 --filter_size 9 \
  --epochs 50 --batch_size 64 --learning_rate 1e-4 \
  --save
```
Trains VQVAE on collected observations using reconstruction loss only (no RL signal). Saved to `./models/MiniGrid-LavaCrossingS9N1-v0/model_{hash}.pt`.

**Phase 3 — PPO with frozen VQVAE** (`model_free/train.py`):
```bash
cd model_free
python train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae --ae_model_version 2 \
  --codebook_size 64 --embedding_dim 64 --filter_size 9 \
  --mf_steps 5000000 --batch_size 4096 \
  --num_envs 16 \
  --ppo_iters 10 --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages --ppo_max_grad_norm 0.5 \
  --ortho_init \
  --model_dir .. \
  --run_name vqvae_pretrain_ppo --device cuda --save
```
**No `--e2e_loss` flag** → encoder frozen from step 0. Loads pretrained VQVAE via hash-matched path. Only policy and critic weights are updated.

**Key differences vs all previous experiments:**

| Aspect | e2e experiments (1–8) | vqvae_pretrain_ppo (11) |
|---|---|---|
| Encoder training | PPO gradients flow through encoder | **Encoder frozen from step 0** |
| Representation source | RL signal shapes the embedding | **Reconstruction loss only** |
| Encoder drift | Inevitable (caused all collapses) | **Impossible** (no gradients) |
| Policy training budget | Shared with encoder training | **Full 5M steps** |
| Risk | Encoder collapses representation | Representation may not be task-relevant |

**Hypothesis:** A reconstruction-trained VQVAE on random-policy observations captures the environment's visual structure (object positions, colors, layout). This is sufficient for PPO to learn a navigation policy because the task-relevant information (agent position, lava position, goal position) is all present in the observations — PPO just needs to learn to read it. With no encoder drift, policy training should be stable throughout all 5M steps, potentially achieving better final performance than any e2e experiment.

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.1927** |
| Final 10-ep avg | **0.0000** |
| Overall avg reward | **0.0102** |
| Best model | `./models/MiniGrid-LavaCrossingS9N1-v0/vqvae_pretrain_ppo_best_model.pt` |
| Final model | `./models/MiniGrid-LavaCrossingS9N1-v0/vqvae_pretrain_ppo_final_model.pt` |

**Analysis — why it failed:**

The hypothesis was falsified. A VQVAE trained purely on reconstruction loss with random-policy data does **not** provide a representation sufficient for PPO to learn navigation. The reasons:

1. **Coverage problem:** Random-policy data almost never reaches the goal (success rate ~1–2% for random walk in LavaCrossing). The VQVAE codebook therefore encodes the distribution of non-goal states. The visual difference between "one step from the goal" and "far from the goal" may be quantized to the same codebook entries, making them indistinguishable to the policy.

2. **Reconstruction loss ≠ task-relevant features:** Reconstruction loss encourages encoding all visual detail equally (wall texture, floor patterns, object colors). Navigation requires encoding *relative position of agent to goal* — a feature that reconstruction loss has no incentive to preserve if it is a small fraction of total pixel variance.

3. **No RL signal to shape the representation:** e2e training, despite its instability, at least produces representations that are causally linked to reward. Frozen reconstruction-trained encoders produce representations that are useful for *reconstructing images*, not for *maximising return*.

**Conclusion:** Offline VQVAE pretraining with reconstruction loss is insufficient. The representation needs either (a) an online RL signal (e2e, but with stabilisation), or (b) a task-aware pretraining objective (e.g. inverse dynamics, reward prediction, or contrastive RL). The e2esnapback approach (peak 0.9988) remains the best result.

---

### 12. vqvae_preinit_snapback_ppo

**Status:** Done
**Script:** `discrete_mbrl/model_free/train.py`
**Hardware:** RTX 4090 (local)
**Log:** `/home/xiar3/experiments/vqvae_preinit_snapback_ppo.log`

**Goal:** Fix poor frozen-pretrain performance by starting from the same pretrained VQVAE but allowing controlled RL finetuning with anti-drift safeguards.

**Key settings:**
- preload pretrained VQVAE: `--ae_model_hash ea136dc75d389f7b850959cd1f78eb6a`
- enable e2e gradients: `--e2e_loss`
- conservative encoder learning: `--encoder_lr 1e-5 --learning_rate 1e-4 --encoder_lr_cosine`
- anti-collapse safety: `--encoder_snapback --snapback_threshold 0.5 --snapback_patience 100 --snapback_min_reward 0.6`
- stable PPO rollout/update settings from strongest runs: `--batch_size 4096 --num_envs 16 --ppo_iters 10 --ppo_batch_size 64 --ppo_gae_lambda 0.95 --ppo_norm_advantages --ppo_max_grad_norm 0.5 --ortho_init`

**Run command:**
```bash
cd discrete_mbrl/model_free
PYTHONPATH=/home/xiar3/experiments/STVqvae:$PYTHONPATH TORCHDYNAMO_DISABLE=1 \
python3 -u train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae --ae_model_version 2 \
  --ae_model_hash ea136dc75d389f7b850959cd1f78eb6a \
  --codebook_size 64 --embedding_dim 64 --filter_size 9 \
  --mf_steps 5000000 --batch_size 4096 --num_envs 16 \
  --ppo_iters 10 --ppo_batch_size 64 --ppo_entropy_coef 0.01 \
  --ppo_gae_lambda 0.95 --ppo_norm_advantages --ppo_max_grad_norm 0.5 \
  --learning_rate 1e-4 --e2e_loss --encoder_lr 1e-5 --encoder_lr_cosine \
  --encoder_snapback --snapback_threshold 0.5 --snapback_patience 100 --snapback_min_reward 0.6 \
  --ortho_init --model_dir .. --run_name vqvae_preinit_snapback_ppo --device cuda --save
```

**Results:**

| Metric | Value |
|---|---|
| Best rolling average reward | **0.9988** |
| Final 10-episode average reward | **0.8988** |
| Overall average reward | **0.9760** |

**Notes:** Controlled finetuning from pretrained VQVAE (low encoder LR + cosine + snapback) fixed the poor frozen-only run and delivered strong final performance.

---

### 13. vae_wm_rl_v1

**Status:** Done
**Pipeline:** `train_encoder.py` → `train_transition_model.py` → `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Logs:** `/home/xiar3/experiments/vae_wm_rl_v1_encoder.log`, `/home/xiar3/experiments/vae_wm_rl_v1_transition.log`, `/home/xiar3/experiments/vae_wm_rl_v1_rl.log`
**Model dir:** `./wm_runs/vae_wm_rl_v1`

**Setup:** VAE encoder (`ae_model_type=vae`, `latent_dim=128`) + continuous transition model + PPO in learned world model (`rl_train_steps=300000`).

**Results:**

| Metric | Value |
|---|---|
| Encoder test loss | **47.3951** |
| Transition test `1_step_state_loss` | **0.8873** |
| Transition test `1_step_reward_loss` | **0.000149** |
| Transition test `1_step_gamma_loss` | **0.00500** |
| World-model PPO train `ep_rew_mean` (peak in log) | **0.0922** |
| World-model PPO final eval (real env) | **0.000 ± 0.000** |

---

### 14. vae_wm_rl_v2

**Status:** Done
**Pipeline:** `train_encoder.py` → `train_transition_model.py` → `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Logs:** `/home/xiar3/experiments/vae_wm_rl_v2_encoder.log`, `/home/xiar3/experiments/vae_wm_rl_v2_transition.log`, `/home/xiar3/experiments/vae_wm_rl_v2_rl.log`
**Model dir:** `./wm_runs/vae_wm_rl_v2`

**Fixes vs v1:**
- Data path fixes to avoid HDF5 per-sample IO stalls (preload path in `ReplayDataset` + shape-probing without full-batch fetch)
- Transition training strengthened: `--trans_epochs 40 --n_train_unroll 8 --e2e_loss`
- RL in world model constrained to shorter horizon: `--rl_unroll_steps 40`
- Longer RL budget: `--rl_train_steps 600000`

**Results:**

| Metric | Value |
|---|---|
| Encoder test loss | **47.2010** |
| Transition test `1_step_state_loss` | **1.6681** |
| Transition test `8_step_state_loss` | **1.8507** |
| Transition test `1_step_reward_loss` | **0.000150** |
| Transition test `1_step_gamma_loss` | **0.00499** |
| World-model PPO train `ep_rew_mean` (max in log) | **0.0699** |
| World-model PPO final eval (real env) | **0.000 ± 0.000** |

**Notes:** Training was stable and fast after IO/data fixes, but policy still failed to transfer from learned world model to real environment.

---

### 15. vqvae_wm_rl_v2_bfs

**Status:** Done
**Pipeline:** `collect_data.py (bfs)` → `train_encoder.py` → `train_transition_model.py` → `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Logs:** `/home/xiar3/experiments/vqvae_wm_rl_v2_bfs_collect.log`, `/home/xiar3/experiments/vqvae_wm_rl_v2_bfs_retry_encoder.log`, `/home/xiar3/experiments/vqvae_wm_rl_v2_bfs_retry_transition.log`, `/home/xiar3/experiments/vqvae_wm_rl_v2_bfs_retry_rl.log`
**Model dir:** `./wm_runs/vqvae_wm_rl_v2_bfs_retry`

**Hypothesis:** Prior world-model runs failed mainly due to reward sparsity in random replay data (only ~0.0095% positive rewards). Rebuilding the buffer with BFS trajectories should increase positive reward coverage and improve transfer.

**Setup:**
- replay collection: `collect_data.py -a bfs -s 200000 -n 8 --env_max_steps 100 --norm_stats`
- encoder: VQVAE (`codebook_size=64`, `embedding_dim=64`, `filter_size=9`)
- transition: discrete model (`trans_epochs=40`, `n_train_unroll=8`)
- RL in world model: PPO with `rl_unroll_steps=20`, `rl_train_steps=600000`

**Results:**

| Metric | Value |
|---|---|
| Replay positive reward rate (BFS data) | **0.017955** (3591 / 200000) |
| Encoder test `quantizer_loss` | **64.2538** |
| Encoder test `recon_loss` | **441.8364** |
| Transition test `1_step_state_loss` | **1.3771** |
| Transition test `1_step_reward_loss` | **0.002698** |
| Transition test `1_step_gamma_loss` | **0.003455** |
| Transition test `8_step_state_loss` | **38.4678** |
| World-model PPO train `ep_rew_mean` (max in log) | **5.56** |
| World-model PPO final eval (real env) | **0.000 ± 0.000** |

**Notes:** Reward density and world-model training signal improved significantly versus random-buffer runs, but real-environment transfer still failed (evaluation remained zero). The most informative failure signal is transition compounding error: `1_step_state_loss = 1.3771` but `8_step_state_loss = 38.4678`. PPO is probably learning to exploit model error over imagined rollouts rather than learning a transferable policy.

---

### 16. vqvae_wm_rl_v3_shortroll_inpolicy

**Status:** Done
**Pipeline:** `collect_data.py (mixed BFS + policy rollouts)` → `train_transition_model.py` → `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vqvae_wm_rl_v3_shortroll_inpolicy`

**Why this experiment:**
- The encoder is likely **not** the main bottleneck anymore. `vqvae_preinit_snapback_ppo` reached **0.8988** final reward in the real environment, so the representation can support good control.
- The world-model pipeline still gets **0.000 real-eval** even when training reward inside the model becomes large (`vqvae_wm_rl_v2_bfs`: world-model PPO train `ep_rew_mean` = **5.56**).
- The most informative failure signal is transition compounding error: in `vqvae_wm_rl_v2_bfs`, `1_step_state_loss = 1.3771` but `8_step_state_loss = 38.4678`. PPO is probably learning to exploit model error over imagined rollouts rather than learning a transferable policy.

**Core hypothesis:**
Real-environment transfer is failing mainly because the transition model is trained off-policy on a replay distribution that does not match the states visited by the learned world-model policy, and because PPO is optimizing over imagined horizons that are already outside the transition model's reliable prediction range.

**Experiment design:**

**Phase 1 — Fix the latent state distribution seen by the transition model**
- Keep the VQVAE encoder **fixed** using the strongest pretrained checkpoint / initialization from the successful real-env run.
- Rebuild the replay buffer using a **mixed dataset**:
  - 20k BFS / solver trajectories
  - 20k rollouts from the best real-environment policy (`vqvae_preinit_snapback_ppo_best_model.pt`)
- Goal: transition training should see successful trajectories and states close to the policy distribution we actually care about.

**Phase 2 — Retrain the transition model for shorter reliable horizons**
- Use the discrete VQVAE transition model as in `vqvae_wm_rl_v2_bfs`.
- Reduce transition unroll target from `n_train_unroll=8` to **`n_train_unroll=4`**.
- Report at least:
  - `1_step_state_loss`
  - `3_step_state_loss`
  - `5_step_state_loss`
  - `10_step_state_loss`
- Success criterion for this phase: multi-step loss should degrade gradually, not explode by 5–8 steps.

**Phase 3 — Constrain PPO to the model's reliable horizon**
- Train PPO in the world model with **`rl_unroll_steps=5`** instead of `20` or `40`.
- Use `rl_train_steps=200000` for the first short-horizon transfer check.
- Evaluate frequently in the **real encoded env**, not just the learned model.

**Planned settings:**

| Component | Setting |
|---|---|
| Encoder | Fixed pretrained VQVAE from strong real-env run |
| Replay data | Mixed BFS + policy rollouts |
| Transition model | Discrete |
| `n_train_unroll` | 4 |
| `rl_unroll_steps` | 5 |
| `rl_train_steps` | 200000 |
| Real-env eval | Required throughout training |

**Results:**

| Metric | Value |
|---|---|
| Mixed replay size | **40,000** (20k BFS + 20k policy rollout) |
| Transition test `1_step_state_loss` | **0.8916** |
| Transition test `2_step_state_loss` | **1.2641** |
| Transition test `3_step_state_loss` | **1.7923** |
| Transition test `4_step_state_loss` | **1.9768** |
| World-model PPO train `ep_rew_mean` (peak seen during run) | **0.263** |
| World-model PPO final eval (real env) | **0.000 ± 0.000** |

**Notes:**
This run achieved the intended transition-model improvement: short-horizon losses stayed controlled instead of exploding the way the previous 8-step model did. However, that improvement still did **not** produce any real-environment transfer. PPO learned something inside the learned world model, but the final real-environment evaluation remained exactly zero.

**Conclusion from this run:**
The bottleneck is still **not PPO tuning**. Even with:
- a proven-good VQVAE encoder,
- mixed replay data closer to the target policy distribution,
- a much better short-horizon transition model,
- and shorter imagined rollouts,

the learned world model still fails to support transferable control.

**Follow-up diagnostic (completed):**
Implemented `discrete_mbrl/diagnose_transition_consistency.py` and evaluated the trained transition model on **25 real-environment episodes** generated by the strong real policy (`vqvae_preinit_snapback_ppo_best_model.pt`). The diagnostic compares:
- **teacher-forced / closed-loop** latent error: predict each next latent from the true current latent
- **open-loop** latent error: roll the transition model forward on its own predicted latents

**Diagnostic results (real-policy trajectories):**

| Horizon | Segments | Teacher-forced state loss | Teacher-forced state acc | Open-loop state loss | Open-loop state acc |
|---|---:|---:|---:|---:|---:|
| 1 | 355 | 474.93 | 0.0188 | 474.93 | 0.0188 |
| 3 | 305 | 484.75 | 0.0203 | 2538.92 | 0.0104 |
| 5 | 255 | 483.94 | 0.0212 | 3821.51 | 0.0110 |
| 10 | 130 | 488.37 | 0.0215 | 4874.16 | 0.0074 |

**Interpretation:**
The transition model is already weak even under teacher forcing on real-policy trajectories, and open-loop latent error compounds sharply with horizon. This explains why PPO in the learned world model can optimize short imagined rollouts yet still fail to transfer to the real environment. The main remaining bottleneck is **transition fidelity along policy-induced trajectories**, not PPO optimization.

**Planner follow-up (completed):**
Implemented `discrete_mbrl/eval_latent_mpc.py` to test whether **replanning from real observations** could salvage performance despite poor long-horizon imagined rollouts. The planner:
- encodes the current real observation,
- brute-forces all action sequences of length 3,
- scores them with the learned transition model,
- executes only the first action,
- and replans from the next real observation (latent MPC / short-horizon MPC).

**Latent MPC result (horizon = 3, 25 real-env episodes):**

| Metric | Value |
|---|---|
| Reward mean | **0.0000** |
| Reward std | **0.0000** |
| Success rate | **0 / 25** |
| Mean episode length | **237.6** |

**Interpretation:**
Short-horizon replanning did **not** recover performance. This means the current transition model is not merely failing because of long open-loop compounding; it is also not accurate enough for useful **local** decision making when used as a planner.

**Stronger transition architecture attempt:**

Tried `transformerdec` as the stronger transition model on the same mixed replay buffer and pretrained VQVAE encoder.

| Metric | Value |
|---|---|
| Transition type | **`transformerdec`** |
| Transition hash | **`b167513bd334cb347350b412e732b828`** |
| Replay buffer | **40,000** (20k BFS + 20k policy rollout) |
| Train setup | **20 epochs, batch size 64, `n_train_unroll=4`** |
| Transition test `1_step_state_loss` | **60.6348** |
| Transition test `2_step_state_loss` | **69.6051** |
| Transition test `3_step_state_loss` | **77.1434** |
| Transition test `4_step_state_loss` | **77.4806** |

**Interpretation:**
This model trained successfully after two code-path fixes:
- Gymnasium discrete-action compatibility in `shared/models/transition_models.py`
- `train_loop` compatibility with trainers that return only `loss_dict`

However, even after converging for 20 epochs, `transformerdec` was still dramatically worse than the current discrete MLP transition model (`0.89 / 1.26 / 1.79 / 1.98` on the same horizons). So the stronger architecture failed the first gate before downstream control.

**Diagnostic follow-up (completed):**

Ran the same real-policy latent consistency diagnostic on the `transformerdec` checkpoint.

| Horizon | Segments | Teacher-forced state loss | Teacher-forced state acc | Open-loop state loss | Open-loop state acc |
|---|---:|---:|---:|---:|---:|
| 1 | 848 | 1036.57 | 0.0045 | 1035.53 | 0.0045 |
| 3 | 798 | 1031.42 | 0.0050 | 1731.43 | 0.0028 |
| 5 | 748 | 1026.16 | 0.0052 | 1838.35 | 0.0026 |
| 10 | 623 | 1017.31 | 0.0053 | 1835.79 | 0.0031 |

**Conclusion:**
`transformerdec` is not a stronger replacement here. It is substantially worse than the discrete baseline both on replay-buffer test loss and on real-policy latent consistency. The next change should be the **state target / supervision** itself, not just the transition architecture.

---

### 17. vae_wm_rl_v3_actionfix_seq

**Status:** Done
**Pipeline:** `train_encoder.py` → `train_transition_model.py` → `train_rl_model.py`
**Hardware:** RTX 4090 (local)

**What changed:** Fixed runner bug so the VAE world-model baseline actually trained sequential `encoder -> transition -> RL` instead of the wrong e2e path.

**Results:**

| Metric | Value |
|---|---|
| Real-env eval | **0.0495 ± 0.2158** |

---

### 18. vae_wm_rl_v4_shortroll

**Status:** Done
**Pipeline:** `train_encoder.py` -> `train_transition_model.py` -> `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vae_wm_rl_v4_shortroll`

**What changed:**
- Kept the regular VAE + continuous transition baseline.
- Matched PPO imagined rollouts to the transition-model training horizon:
  - `n_train_unroll = 8`
  - `rl_unroll_steps = 8`
  - `rl_train_steps = 300000`
- Used the current default 40k replay buffer (`20k` BFS + `20k` policy rollouts).

**Results:**

| Metric | Value |
|---|---|
| Encoder test loss | **56.9123** |
| Transition test `1_step_state_loss` | **0.8339** |
| Transition test `4_step_state_loss` | **0.8168** |
| Transition test `8_step_state_loss` | **0.8169** |
| Real-env RL eval | **0.2486 ± 0.4306** |
| Real-env mean episode length | **42.8** |

**Conclusion:**
- Matching PPO horizon to the learned model helped materially: real-env reward improved from `0.0495 ± 0.2158` in `vae_wm_rl_v3_actionfix_seq` to `0.2486 ± 0.4306`.
- That still leaves the policy weak. The remaining failure mode looks like a combination of:
  - transition capacity limits in the continuous world model, and
  - PPO overfitting to a short surrogate task when trained at a single fixed imagined horizon.

---

### 19. vae_wm_rl_v5_curriculum

**Status:** Done
**Pipeline:** `train_encoder.py` -> `train_transition_model.py` -> `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vae_wm_rl_v5_curriculum`

**Why this experiment:**
- `vae_wm_rl_v4_shortroll` showed that reducing imagined horizon helps, but a fixed `8`-step world-model task still undertrains the policy for the longer real task.
- The continuous transition model is also still fairly small by default (`256 x 3` MLP), which likely leaves avoidable latent prediction error on policy-relevant trajectories.

**Changes in this run:**
- Stronger transition model:
  - `trans_hidden = 512`
  - `trans_depth = 5`
  - `trans_epochs = 80`
  - `n_train_unroll = 12`
- RL curriculum in the world model:
  - stage 1: `rl_unroll_steps = 8`, `100k` PPO steps
  - stage 2: `rl_unroll_steps = 12`, `100k` PPO steps
  - stage 3: `rl_unroll_steps = 16`, `100k` PPO steps
- Code change:
  - `train_rl_model.py` now supports staged PPO horizons via `--rl_stage_unrolls` and `--rl_stage_steps`.

**Hypothesis:**
- Start PPO inside the model's most reliable horizon, then extend gradually so the policy does not overfit to an 8-step surrogate objective.
- Give the transition model enough capacity to keep latent rollouts coherent deeper into the episode.

**Results:**

| Metric | Value |
|---|---|
| Encoder test loss | **60.8914** |
| Transition test `1_step_state_loss` | **0.8245** |
| Transition test `4_step_state_loss` | **0.8233** |
| Transition test `8_step_state_loss` | **0.8247** |
| Transition test `12_step_state_loss` | **0.8215** |
| Real-env RL eval | **0.0000 ± 0.0000** |
| Real-env mean episode length | **467.5** |

**Notes:**
- The stronger transition model did improve latent test losses slightly relative to `vae_wm_rl_v4_shortroll`.
- PPO also learned a stronger imagined policy inside the world model by the end of stage 3 (`ep_rew_mean` around `0.63` with `ep_len_mean = 16`).
- That still did **not** transfer: the final real-environment evaluation was exactly zero.

**Conclusion:**
- For this VAE baseline, better transition capacity plus staged horizon extension was **not enough** to solve the transfer problem.
- The remaining issue is likely deeper than simple horizon mismatch: either the VAE latent state is still too weak for control-relevant rollout fidelity, or PPO is still exploiting model inaccuracies even after the curriculum.

---

### 20. vae_wm_rl_v9_transfer_gap

**Status:** Done
**Pipeline:** Post hoc diagnostic on saved `v5` artifacts
**Hardware:** CPU (local sandbox)
**Run dir:** `./wm_runs/vae_wm_rl_v9_transfer_gap`

**Why this diagnostic was needed:**
- `vae_wm_rl_v5_curriculum` showed a strong imagined PPO policy but zero real transfer.
- The earlier `v6_diag` script only measured latent rollout MSE and used the shared PPO alias, so it could not separate "weak world model" from "policy exploiting model-specific reward drift".
- `v5` did not preserve a per-run PPO checkpoint under its run directory, so this diagnostic uses the shared policy alias `discrete_mbrl/models/MiniGrid-LavaCrossingS9N1-v0/ppo_world_model.zip` as the closest available artifact. Its file timestamp still matches the `v5` era, but treat this as a best-effort post hoc diagnosis rather than a perfect replay.

**What this diagnostic measured:**
- Real-environment evaluation of the saved PPO policy on encoded observations.
- World-model evaluation of that same PPO policy with imagined horizons `8`, `12`, `16`, and `500`.
- Teacher-forced one-step transition / reward / gamma error on real-policy trajectories.
- Open-loop multi-step rollout bias on those same real-policy action sequences.

**Results:**

| Metric | Value |
|---|---|
| Real-env eval | **0.0397 ± 0.1191** |
| Real-env mean episode length | **92.8** |
| World-model eval @ 8 | **0.2111 ± 0.1775** |
| World-model eval @ 12 | **0.3304 ± 0.4165** |
| World-model eval @ 16 | **0.4848 ± 0.3208** |
| World-model eval @ 500 | **18.3469 ± 21.9467** |
| Teacher-forced 1-step state MSE | **0.8307** |
| Teacher-forced 1-step reward abs error | **0.0236** |
| Teacher-forced predicted reward mean | **0.0235** |
| Teacher-forced actual reward mean | **0.0000** |
| Open-loop 16-step state MSE | **0.8537** |
| Open-loop 16-step predicted return mean | **0.2686** |
| Open-loop 16-step actual return mean | **0.0000** |
| Open-loop 16-step continuation bias | **-0.2746** |

**Interpretation:**
- The main failure is **not** just local one-step instability. One-step teacher-forced errors are only slightly noisy, but the reward head already assigns positive reward on average to real-policy states that produce zero real reward.
- The much larger problem is **compounding rollout optimism**: along real action sequences, the model predicts increasingly positive multi-step return even though the actual return stays exactly zero. That gap grows monotonically with rollout horizon (`0.0235` at 1 step -> `0.2686` at 16 steps).
- The policy is therefore learning to exploit **imagined reward drift** in model rollouts, not to solve the real task. The huge `500`-step imagined reward (`18.35`) versus near-zero real reward is the clearest signal.
- Gamma prediction is not the main optimism source here. If anything, the model is slightly pessimistic about continuation on real trajectories. The dominant issue is reward / state drift into fictitious high-value latent regions.

**Conclusion:**
- `vae_wm_rl_v5_curriculum` failed primarily because PPO found a shortcut in the learned continuous world model: multi-step imagined rollouts create non-real reward, and the policy overfits to that artifact.
- Improving transition capacity alone was insufficient because it did not remove the reward-bias channel that PPO exploited.

---

### 21. vae_wm_rl_v11_conservative_reward_longshort

**Status:** Running
**Pipeline:** `train_encoder.py` -> `train_transition_model.py` -> `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vae_wm_rl_v11_conservative_reward_longshort`

**Why this experiment:**
- `v10` sharply reduced short-horizon imagined reward hallucination, but real transfer still stayed at zero.
- That leaves two plausible blockers:
  - PPO no longer has enough useful learning signal after reward optimism is suppressed, or
  - latent rollout error still breaks control once the policy is pushed beyond the most reliable horizon.

**Changes in this run:**
- Keep the same conservative reward transition penalties from `v10`:
  - `--trans_reward_overestimate_coef 2.0`
  - `--trans_reward_zero_target_coef 4.0`
  - `--trans_reward_zero_margin 0.0`
- Keep periodic real-environment evaluation and best-checkpoint selection.
- Change only the PPO curriculum to spend much longer at the shortest reliable horizon:
  - `rl_stage_unrolls = 8,8,12,16`
  - `rl_stage_steps = 200k,200k,100k,100k`
  - total PPO steps = `600k`

**Hypothesis:**
- If `v10` failed mainly because PPO lost a usable optimization signal once reward hallucination was removed, then giving PPO much more budget at horizon `8` should recover a transferable policy before extending to longer imagined horizons.
- If real transfer still remains zero after this schedule, the remaining blocker is more likely state/control fidelity rather than reward optimism alone.

---

### 22. vae_encoder_frozen_ppo_v1

**Status:** Running
**Pipeline:** Frozen-encoder real-environment PPO
**Hardware:** RTX 4090 (local)
**Encoder source:** `./wm_runs/vae_wm_rl_v11_conservative_reward_longshort`
**Run script:** `run_vae_encoder_frozen_ppo_v1.sh`

**Why this experiment matters:**
- `v11` recovered some world-model transfer, but that still does not tell us whether the VAE encoder itself is strong enough for control.
- The cleanest test is to remove the world model entirely and train PPO directly in the real environment on top of the frozen `v11` encoder.

**Setup:**
- Load the `v11` VAE encoder checkpoint:
  - `ae_model_hash = 393f184899f1c7bd6740a092b342902c`
- Keep the encoder frozen from step 0:
  - no `--e2e_loss`
- Train PPO directly in the real environment with the stable vectorized setup:
  - `num_envs = 16`
  - `batch_size = 4096`
  - `ppo_batch_size = 256`
  - `ppo_gae_lambda = 0.95`
  - `ppo_max_grad_norm = 0.5`
  - `ortho_init = True`
  - `mf_steps = 5,000,000`

**Question:**
- Can the frozen `v11` VAE representation support strong real-environment PPO on its own?

**Interpretation rule:**
- If this run gets strong reward, the encoder is good enough and the remaining problem is the world model.
- If it stays weak, the encoder is still a major bottleneck.

---

### 23. encoder_localctx_vqvae_v1

**Status:** Running
**Pipeline:** `train_encoder.py` -> `validate_encoder_checkpoint.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/encoder_localctx_vqvae_v1`

**Why this experiment exists:**
- The frozen `v11` VAE encoder PPO baseline only reached a transient reward around `0.30` and collapsed back to zero by the end.
- That means the current VAE representation is not strong enough to treat the encoder as solved.
- Before more world-model work, we need a stronger encoder trained and validated on its own.

**Encoder choice:**
- `local_ctx_vqvae` instead of plain VAE.
- Reason:
  - the repo already has evidence that VQ-style encoders can support strong control,
  - `local_ctx_vqvae` adds a context path for better reconstruction while keeping a discrete bottleneck,
  - the trainer already includes safeguards against context-bypass collapse.

**Training setup:**
- `ae_model_type = local_ctx_vqvae`
- `embedding_dim = 128`
- `codebook_size = 256`
- `filter_size = 9`
- `ctx_channels = 128`
- `ctx_aux_coef = 2.0`
- `entropy_penalty_coef = 0.05`
- `code_dropout_rate = 0.15`
- `mae_mask_ratio = 0.5`
- `epochs = 40`

**Validation setup:**
- No transition model.
- No PPO.
- Run encoder-only validation on the saved checkpoint:
  - test reconstruction / quantizer losses,
  - agent-region vs background reconstruction stats,
  - codebook usage / entropy / perplexity.

**Question:**
- Can we train a representation that is visibly healthier than the current VAE before involving control at all?

---

### 24. vqvae_wm_compact_v1

**Status:** Running
**Pipeline:** `full_train_eval.py` (encoder → transition → RL)
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vqvae_wm_compact_v1`
**Run script:** `run_vqvae_wm_compact_v1.sh`

**Core hypothesis:**
All previous world-model experiments used `filter_size=9`, producing **81 spatial tokens** (9×9 grid) from the VQVAE encoder. The transition model must predict 81 categorical distributions simultaneously — a prediction target so large that even 1-step teacher-forced accuracy on policy trajectories was catastrophic (loss ~475 in experiment 16 diagnostic). Reducing to `filter_size=3` (3×3 = **9 tokens**) makes the transition model's job 9× simpler, while increasing `codebook_size` (64→256) and `embedding_dim` (64→128) preserves representational capacity.

**Key changes vs vqvae_wm_rl_v2_bfs (experiment 15) and v3_shortroll_inpolicy (experiment 16):**

| Parameter | Previous | This run | Rationale |
|---|---|---|---|
| `filter_size` | 9 (81 tokens) | **3 (9 tokens)** | 9× simpler prediction target |
| `codebook_size` | 64 | **256** | Richer vocabulary to compensate for fewer spatial positions |
| `embedding_dim` | 64 | **128** | Richer per-token features |
| `trans_hidden` | 256 | **512** | Stronger transition model |
| `trans_depth` | 3 | **5** | Deeper transition MLP |
| `trans_epochs` | 40 | **80** | More training for the new latent space |
| `encoder epochs` | ~20 | **50** | More encoder training for new architecture |
| `n_train_unroll` | 4–8 | **8** | Multi-step training horizon |
| `rl_unroll_steps` | 5–20 | **5** | Short horizon within model's reliable range |
| `rl_train_steps` | 200k–600k | **600000** | Full PPO budget |
| `rl_eval_freq` | 0 | **5000** | Frequent real-env evaluation |

**State dimensions comparison:**

| | Previous (filter_size=9) | This run (filter_size=3) |
|---|---|---|
| Spatial tokens | 81 | **9** |
| Codebook entries | 64 | **256** |
| Embedding dim | 64 | **128** |
| Flat state dim | 5184 | **1152** |
| Transition output | 81 × 64 = 5184 logits | **9 × 256 = 2304 logits** |

**Success criteria:**
1. Transition model multi-step losses should degrade gradually (not explode like 1.38 → 38.47 in experiment 15)
2. Real-env PPO eval should be > 0 (any transfer is progress vs 0.000 in all prior world-model runs)
3. Stretch goal: real-env eval > 0.2 (matching vae_wm_rl_v4_shortroll, experiment 18)

**Results:**

| Metric | Value |
|---|---|
| Encoder `quantizer_loss` | **22.54** |
| Encoder `recon_loss` | **1490.8** |
| Transition `1_step_state_loss` | **0.2142** |
| Transition `8_step_state_loss` | **0.2142** (perfectly flat — zero compounding error ✅) |
| World-model PPO `ep_rew_mean` (peak) | **0.146** (PPO barely learning inside model) |
| Real-env eval (97 evals × 25 eps) | **0.000 ± 0.000** |

**Diagnosis:**

The transition model problem is solved — 8-step loss equals 1-step loss exactly, unlike experiment 15 (1.38 → 38.47). However two new problems emerged:

1. **Representation too coarse:** filter_size=3 gives each token a 3×3 grid-cell footprint. The agent (1 cell) almost never crosses a token boundary on a single step, so the encoded state looks static. The transition model learned to predict "no change" everywhere — trivially correct but useless for planning.

2. **Imagined rollouts too short:** With `rl_unroll_steps=5` and only 1.8% positive reward rate in replay data, expected imagined reward per episode = 5 × 0.018 = **0.09**. PPO had almost no reward signal to learn from even inside the model.

---

### 25. vqvae_wm_compact_v2

**Status:** Running
**Pipeline:** `full_train_eval.py` (encoder → transition → RL)
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vqvae_wm_compact_v2`
**Run script:** `run_vqvae_wm_compact_v2.sh`
**Log:** `/home/xiar3/experiments/vqvae_wm_compact_v2.log`

**Changes vs compact_v1 (experiment 24):**

| Parameter | compact_v1 | compact_v2 | Rationale |
|---|---|---|---|
| `filter_size` | 3 (9 tokens) | **5 (25 tokens)** | ~1.8 grid cells/token — most single-step agent moves cross a token boundary |
| `rl_unroll_steps` | 5 | **20** | Longer imagined episodes give PPO more steps to encounter the goal; safe because compact_v1 proved flat transition loss at all horizons |

**State dimensions:**

| | compact_v1 (fs=3) | compact_v2 (fs=5) |
|---|---|---|
| Spatial tokens | 9 | **25** |
| Grid cells per token | 3.0 × 3.0 | **1.8 × 1.8** |
| MLP input dim | 1,159 | **3,207** |
| MLP output logits | 2,304 | **6,400** |

**Results:** Completed. Encoder and transition model trained identically to compact_v1 design but with better representation. Real-env evals: **0.000 across all 97 eval checkpoints** (5k-step intervals over 600k RL steps).

**Transition model diagnostics** (`diagnose_compact_v2.py`, policy_20k buffer, 200 episodes × 100 steps, all zero-reward):

| Horizon | TF loss | TF acc  | OL loss | OL acc |
|---------|---------|---------|---------|--------|
| 1       | 0.3792  | 0.9998  | 0.3792  | 0.9998 |
| 3       | 0.3791  | 0.9998  | 2.3242  | 0.9792 |
| 5       | 0.3792  | 0.9998  | 7.9409  | 0.9410 |
| 10      | 0.3792  | 0.9998  | 23.5861 | 0.8597 |

Reward on policy_20k buffer (no goal transitions): `pred_mean=0.002824`, `pred_max=0.082177` — near zero.

**Root cause analysis:**
- Teacher-forced accuracy is near-perfect (99.98%) because the model learned "predict same state" — correct for the many wall-bumping transitions in the policy_20k buffer.
- Open-loop error compounds heavily: at horizon 10 the loss is 62× worse than teacher-forced; at horizon 20 it would be far worse.
- `rl_unroll_steps=20` was the wrong fix. PPO learned to navigate hallucinated states in corrupted 20-step imagined rollouts. The `ep_rew_mean=0.5–0.6` in imagined world was from states that do not correspond to real-env dynamics. The transition model's `DiscreteTransitionTrainer` does use open-loop (autoregressive) training (line 926: `encodings = next_logits_pred.argmax(dim=1).detach()`), but the compounding error still grows faster than training corrects it.
- Reliable open-loop horizon is ≤3 steps (OL acc 97.9%). At 5 steps: 94.1% (marginal). At 20 steps: likely < 70%.

---

### 26. vqvae_wm_compact_v3

**Status:** Running
**Pipeline:** `full_train_eval.py` (encoder → transition → RL)
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vqvae_wm_compact_v3`
**Run script:** `run_vqvae_wm_compact_v3.sh`
**Log:** `/home/xiar3/experiments/vqvae_wm_compact_v3.log`

**Change vs compact_v2 (one targeted fix):**

| Parameter | compact_v2 | compact_v3 | Rationale |
|---|---|---|---|
| `rl_unroll_steps` | 20 | **5** | At horizon 5 the transition model has 94.1% OL token accuracy (7.94 OL loss vs 0.38 TF). At horizon 20 states are heavily corrupted. PPO must stay within reliable prediction horizon. |

All other parameters unchanged: `filter_size=5`, `codebook_size=256`, `embedding_dim=128`, `trans_hidden=512`, `trans_depth=5`, `n_train_unroll=8`, `rl_train_steps=600k`.

**Rationale for reward signal:** The training buffer includes BFS data with goal-reaching transitions. Imagined rollouts initialised from states 1–5 steps from the goal will encounter the goal within 5 imagined steps, providing a real reward signal in the reliable prediction zone.

**Results:** Killed at 244k/600k. 39 real-env evals, all 0.000. Same pattern as compact_v2 — even reducing rollout length to 5 doesn't fix the fundamental problem.

---

### 27. vqvae_wm_dyna_v1

**Status:** Done — Failed (0.000 real-env transfer)
**Pipeline:** `full_train_eval.py` (encoder → transition → **Dyna RL**)
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vqvae_wm_dyna_v1`
**Run script:** `run_vqvae_wm_dyna_v1.sh`

**Architectural change: Dyna-style training**

All prior world-model experiments (13–26) trained PPO in **pure imagination** — the policy never saw a single real reward during training. This is the root cause of 0.000 transfer: the policy optimised for hallucinated rewards in divergent imagined states.

Dyna fixes this by training PPO on a **mixed VecEnv** of real and imagined environments simultaneously:

| Component | Count | Purpose |
|---|---|---|
| Real envs (`ObsEncoderWrapperGymnasium`) | 4 | Provide grounded reward signal from actual MDP |
| Imagined envs (`PredictedModelWrapperGymnasium`) | 8 | Data augmentation via short-horizon transition model rollouts |
| Imagined horizon (`dyna_horizon`) | 3 steps | Within 97.9% open-loop accuracy zone |

SB3's PPO collects rollouts from all 12 envs each update. Real envs provide sparse-but-true rewards. Imagined envs provide 2× more experience per update.

After Dyna training (2M steps), a Phase 2 finetunes on pure real envs (500k steps).

**Why this should work:**
- Model-free PPO with frozen VQVAE (exp 12) achieved 0.9988 in 5M steps. The 4 real envs in Dyna replicate this proven approach.
- The 8 imagined envs add data augmentation that should improve sample efficiency.
- Even if imagined data adds zero value, the real envs alone will eventually solve the task.

**Key settings:**

| Parameter | Value | Rationale |
|---|---|---|
| `--dyna` | enabled | Mixed real+imagined training |
| `--n_real_envs` | 4 | Real MDP interactions for grounded reward |
| `--n_imagined_envs` | 8 | 2:1 imagined:real ratio for data augmentation |
| `--dyna_horizon` | 3 | Within reliable OL zone (97.9% accuracy at h=3) |
| `--rl_train_steps` | 2,000,000 | 2M total steps across 12 envs |
| `--rl_finetune_steps` | 500,000 | Phase 2: pure real-env finetuning |
| `--rl_eval_freq` | 10,000 | Monitor real-env transfer throughout |

Encoder and transition model: identical to compact_v2/v3 (filter_size=5, codebook_size=256, etc.)

**Results:**

| Phase | Steps | Outcome |
|---|---|---|
| Dyna Phase 1 | 3,000,000 | 0.000 real-env eval throughout — policy adapted to imagined rewards but never transferred |
| Phase 2 (real finetune) | 3,000,000 | Policy collapse — ep_rew_mean 0.15 → 0.000, episodes stuck at 500-step timeout |

**Post-mortem:** Dyna Phase 1 produced negative transfer: the policy learned to navigate imagined environments (hallucinated states with inaccurate rewards) and failed to transfer to reality. Phase 2 switched to pure real envs using `set_env()` with the Dyna-trained weights — these weights were adapted to imagined rewards and collapsed immediately. The imagined data did not add useful augmentation; instead it poisoned the policy against real-environment behavior.

---

### 28. mf_frozen_vqvae_v2

**Status:** Done — Failed (0.000 real-env transfer)
**Pipeline:** `full_train_eval.py` (encoder + frozen encoder PPO)
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/mf_frozen_vqvae_v2`
**Run script:** `run_mf_frozen_vqvae_v2.sh`

**Goal:** Return to proven model-free approach. Designed to replicate exp #12 (0.9988) using the compact_v2 encoder (filter_size=5, 25 tokens, codebook_size=256) with a truly frozen encoder + SB3 PPO via the Dyna infrastructure with n_imagined_envs=0.

**Key settings:**
- `--dyna --n_real_envs 24 --n_imagined_envs 0` — 24 real parallel envs, no imagined
- `--trans_epochs 1` — minimal transition training (not used)
- `--rl_train_steps 5000000` — 5M total steps
- `--ppo_n_steps 1024` — rollout length per env
- Encoder truly frozen (part of env wrapper, no gradients)

**Results:**

| Metric | Value |
|---|---|
| Best eval reward | ~0.000 (first eval at 73k steps) |
| Final eval reward | 0.000 |
| Total steps completed | 5,000,000 |
| Runtime | ~1h 40m |

**Post-mortem (root cause analysis):**

Experiment #28 replicated **exp #11** (frozen VQVAE + PPO, 0.1927 best / 0.0000 final), NOT exp #12.

Exp #12 succeeded because it used **e2e gradients** with very low encoder LR (`1e-5`) + cosine decay + snapback. The encoder was pretrained but NOT frozen — it was slowly fine-tuned by the RL signal.

Frozen VQVAE invariably fails because:
1. **Coverage problem**: Random-policy replay data achieves ~1-2% goal success. VQVAE codebook encodes non-goal states. "One step from goal" and "far from goal" may quantize to identical tokens.
2. **Reconstruction ≠ navigation**: Reconstruction loss encodes all visual detail equally; navigation requires relative agent-to-goal position, which reconstruction has no incentive to preserve.
3. **No RL signal**: Frozen encoders never develop task-relevant representations.

Additionally, the compact_v2 encoder (25 tokens vs 81 in exp #12) has less spatial resolution, making these problems worse.

### 29. mf_e2e_snapback_v1

**Status:** Done
**Script:** `discrete_mbrl/model_free/train.py`
**Hardware:** RTX 4090 (local)
**Run script:** `run_mf_e2e_snapback_v1.sh`
**Log:** `wm_runs/mf_e2e_snapback_v1.log`

**Goal:** Exact replication of exp #12 (vqvae_preinit_snapback_ppo, 0.9988 peak / 0.8988 final) on current hardware to confirm approach still works. Then explore compact_v2 encoder variant if successful.

**Root cause of exp 28 failure:**
Experiments 11 and 28 used truly frozen VQVAE encoders trained on reconstruction loss. These encoders:
- Trained on random-policy data with ~1-2% goal coverage
- Encode all visual detail equally (no task-relevant bias)
- Cannot be fine-tuned by RL gradient → policy must use representations that never adapt

What actually worked (exp #12):
- Same pretrained VQVAE as starting point
- e2e gradients: PPO gradients flow through the encoder at 1/10 the policy LR
- Cosine decay: encoder LR anneals from 1e-5 → 0 (quasi-freezes late in training)
- Snapback: if reward drops >50% from peak (after first reaching 0.6), restore best encoder state and hard-freeze permanently

**Key settings (identical to exp #12):**

| Parameter | Value |
|---|---|
| Encoder | Pretrained VQVAE (hash ea136dc75d389f7b850959cd1f78eb6a) |
| Architecture | filter_size=9, codebook_size=64, embedding_dim=64 (81 tokens) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `num_envs` | 16 |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 (10× lower) |
| `encoder_lr_cosine` | 1e-5 → 0 over 1221 batches |
| `e2e_loss` | enabled |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.6 |
| `ortho_init` | enabled |

**Results:** Peak 0.9988, Final 0.9984, Overall 0.9732. Exact replication of exp #12 — confirmed. No snapback triggered; encoder remained stable throughout.

---

### Experiment 30: mf_e2e_wm_aux_v1

**Status:** Done

**Script:** `run_mf_e2e_wm_aux_v1.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** Adding a world-model auxiliary predictive loss to the proven e2e PPO approach (exp #29) will improve sample efficiency and/or final performance by shaping the encoder to learn dynamics-aware features.

**Approach:** A discrete transition model is trained online (on real PPO rollout data) to predict next-state tokens from current-state + action. The prediction loss (cross-entropy on discrete latent indices) is added to the PPO objective with coefficient `wm_aux_coef=0.1`. Gradients flow back through the VQVAE encoder via straight-through estimator. The transition model is also independently trained via its own optimizer to stay on-distribution as the encoder evolves.

**Key changes from exp #29:**
- `--use_world_model` enables online world model training + auxiliary loss
- `--wm_aux_coef 0.1` controls auxiliary loss weight in PPO objective
- `--wm_train_freq 1` trains transition model every PPO batch
- `--trans_model_type discrete --trans_hidden 256 --trans_depth 3` (2.9M params)

| Parameter | Value |
|---|---|
| Encoder | Pretrained VQVAE (hash ea136dc75d389f7b850959cd1f78eb6a) |
| Architecture | filter_size=9, codebook_size=64, embedding_dim=64 (81 tokens) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `num_envs` | 16 |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 (10× lower) |
| `encoder_lr_cosine` | 1e-5 → 0 over 1221 batches |
| `e2e_loss` | enabled |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.6 |
| `ortho_init` | enabled |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `wm_train_freq` | 1 |
| `trans_model_type` | discrete (256×3, 2.9M params) |
| `trans_learning_rate` | 1e-3 |

**Results:** Peak **0.9988**, Final **0.9985**, Overall avg **0.9620**. Essentially identical to the baseline (exp #29). The auxiliary predictive loss did not hurt stability — no snapback triggered, encoder remained stable throughout. However, the slightly lower overall average (0.9620 vs 0.9732) suggests marginally slower convergence, likely due to the optimizer balancing the extra loss term early in training. The world model auxiliary loss at this coefficient (0.1) is neutral: neither a clear benefit nor a detriment.

---

### Experiment 31: mf_e2e_wm_aux_v2

**Status:** Done

**Script:** `run_mf_e2e_wm_aux_v2.sh` → `discrete_mbrl/model_free/train.py`

**Change from exp #30:** Remove the redundant standalone transition model training step (`--wm_standalone_train` omitted, default=False). In exp #30 the transition model was updated twice per batch — once via the aux loss gradient inside PPO's backward pass, and once via its own separate Adam optimizer. The standalone step is redundant because: (1) the encoder barely moves (lr 1e-5 → 0), so no moving target to chase; (2) the trans model is already updated through PPO. This experiment keeps only the aux loss in PPO.

| Parameter | Value |
|---|---|
| Encoder | Pretrained VQVAE (hash ea136dc75d389f7b850959cd1f78eb6a) |
| Architecture | filter_size=9, codebook_size=64, embedding_dim=64 (81 tokens) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `e2e_loss` | enabled |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.6 |
| `ortho_init` | enabled |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `wm_standalone_train` | **False** (omitted — key difference from exp #30) |
| `trans_model_type` | discrete (256×3, 2.9M params) |
| `trans_learning_rate` | 1e-3 (used only if standalone train were enabled) |

**Results:** Peak **0.9988**, Final **0.9986**, Overall avg **0.9681**. Removing the redundant standalone training step slightly improved both final reward and overall average vs exp #30 (0.9985 / 0.9620), closing the gap with the baseline (exp #29: 0.9984 / 0.9732). All three experiments are essentially tied at the peak. The world model auxiliary loss remains neutral at coef=0.1 on this environment.

---

### Experiment 32: mf_e2e_wm_doorkey_v1

**Status:** Done

**Script:** `run_mf_e2e_wm_doorkey_v1.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** The world model auxiliary loss (neutral on LavaCrossing) will show a measurable benefit on `MiniGrid-DoorKey-8x8-v0`. DoorKey requires a multi-step sub-goal sequence — find key → pick up key → navigate to door → open door → reach goal — with sparse reward delivered only at episode end. This longer credit assignment horizon and richer transition structure are exactly the conditions under which forcing the encoder to predict next-state tokens should produce dynamics-aware features that help the policy learn faster and reach higher final performance.

**Key changes from exp #31:**
- `--env_name` changed to `MiniGrid-DoorKey-8x8-v0`
- `--ae_model_hash` **omitted** — no pretrained VQVAE exists for DoorKey; encoder is randomly initialised and fine-tuned e2e alongside the policy
- `--snapback_min_reward` lowered from `0.6` → `0.1` — DoorKey is harder so the snapback guard should activate earlier, once any non-trivial reward appears

| Parameter | Value |
|---|---|
| **Environment** | **MiniGrid-DoorKey-8x8-v0** |
| Encoder | Random init VQVAE v2 (no pretrained hash) |
| Architecture | filter_size=9, codebook_size=64, embedding_dim=64 (81 tokens) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `e2e_loss` | enabled |
| `encoder_snapback` | threshold=0.5, patience=100, **min_reward=0.1** |
| `ortho_init` | enabled |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `wm_standalone_train` | False |
| `trans_model_type` | discrete (256×3, 2.9M params) |

**Results:** Peak **0.9988**, Final **0.9984**, Overall avg **0.9979**. No snapback triggered; encoder remained stable throughout. The world model auxiliary loss on DoorKey-8x8 produced the **highest overall average across all experiments** (0.9979 vs 0.9732 for the best LavaCrossing run), despite starting from a randomly initialised VQVAE — no pretrained encoder. The harder multi-step sub-goal structure (find key → pick up → open door → reach goal) appears to benefit from the world model's dynamics-aware shaping, consistent with the hypothesis. Peak matched at batch ~361 (~30% through training), indicating fast convergence.

---

### Experiment 33: mf_e2e_semantic_probe

**Status:** Done

**Script:** (Phase 1) `run_semantic_probe.sh` → `discrete_mbrl/probe_semantics.py` | (Phase 2) `run_mf_e2e_semantic_aux.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** The 9×9 VQVAE spatial grid has a direct 1:1 correspondence with the MiniGrid cell grid (72×72 image → 9×9 latent via `AdaptiveAvgPool2d(9)` → each latent position covers one 8×8-pixel tile). This means each codebook token can, in principle, encode the identity of the object at that cell. A small auxiliary head that predicts per-position object type (cross-entropy over 11 MiniGrid object classes) will (a) diagnose how much semantic content the frozen pretrained VQVAE already captures, and (b) when added as an aux loss during e2e training, explicitly force each spatial code to be semantically distinct — preventing the codebook from conflating objects like lava and empty floor that may have similar reconstruction loss.

**Two-phase design:**

**Phase 1 — Frozen probe (diagnostic, no RL):**
Load pretrained VQVAE (hash `ea136dc75d389f7b850959cd1f78eb6a`), freeze all weights. Collect ~50k frames with a random policy. At each step, record the 81-position quantized embeddings (64-dim each) and the ground-truth object type per cell from `env.unwrapped.grid.encode()[:,:,0]` with agent overlaid. Train a single `nn.Linear(64, 11)` probe (one shared probe applied independently per position) via SGD + cross-entropy. Report per-class accuracy and confusion matrix.

This is fast (CPU-feasible, no RL) and answers: *does the reconstruction-trained codebook already encode object identity at each cell?*

**Phase 2 — Semantic aux loss during e2e PPO:**
Add a small `SemanticHead` (`Linear(64→64→ReLU→11)`) trained jointly with PPO. At each minibatch update, reshape the continuous quantized embeddings from `(B, 5184)` → `(B, 81, 64)`, run through the head → `(B, 81, 11)` logits, compute cross-entropy against ground-truth object-type labels collected alongside observations, and add `sem_aux_coef * sem_loss` to the total loss. Gradient flows through STE back to the encoder, nudging each spatial code toward semantic distinctiveness.

A `SemanticLabelWrapper` in `env_helpers.py` exposes `grid.encode()[:,:,0]` (with agent overlay) as a side channel alongside the RGB observation. The PPO `ReplayBuffer` and `train.py` batch collection store these labels as an extra key `semantic_grid` — same pattern as `next_obs` was added for the world model aux loss.

**Key changes from exp #31:**
- `--use_semantic_aux` enables the semantic label wrapper + SemanticHead
- `--sem_aux_coef 0.05` (half the world model coefficient — conservative start)
- `--sem_head_hidden 64` (small MLP, will not compete with PPO)
- World model aux kept (`--use_world_model`, `--wm_aux_coef 0.1`) for additive comparison

| Parameter | Value |
|---|---|
| Environment | MiniGrid-LavaCrossingS9N1-v0 |
| Encoder | Pretrained VQVAE (hash ea136dc75d389f7b850959cd1f78eb6a) |
| Architecture | filter_size=9, codebook_size=64, embedding_dim=64 (81 tokens) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `e2e_loss` | enabled |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.6 |
| `ortho_init` | enabled |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `wm_standalone_train` | False |
| `trans_model_type` | discrete (256×3, 2.9M params) |
| **`use_semantic_aux`** | **enabled** |
| **`sem_aux_coef`** | **0.05** |
| **`sem_head_hidden`** | **64** |

**Semantic label mapping (MiniGrid OBJECT_TO_IDX):**
`unseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10`
LavaCrossing uses: empty, wall, lava, goal, agent (5 of 11 classes). DoorKey would add door and key, making the semantic head more informative.

**Implementation touch points:**
1. `env_helpers.py` — `SemanticLabelWrapper`: wraps any MiniGrid env, reads `env.unwrapped.grid.encode()[:,:,0]` (9×9 int array), overlays agent at `agent_pos` with type=10, returns it as `obs['semantic_grid']`
2. `shared/models/transition_models.py` (or new `semantic_head.py`) — `SemanticHead(nn.Module)`: `Linear(64,64) → ReLU → Linear(64,11)`, applied to each of 81 positions independently
3. `model_free/train.py` — pass `sem_aux_coef` and `use_semantic_aux` args to `PPOTrainer`; store `semantic_grid` in batch alongside `next_obs`
4. `model_free/ppo.py` — in `update()`, after world model aux block: reshape `cont_states → (B,81,64)`, forward through `SemanticHead`, cross-entropy vs `minibatch['semantic_grid']`, add to `total_loss`
5. `probe_semantics.py` — standalone Phase 1 script (~150 lines): load VQVAE, random rollouts, collect `(embed, label)` pairs, train linear probe, print accuracy + save confusion matrix

**Comparison baseline:**

| Experiment | Aux losses | Peak | Final | Overall avg |
|---|---|---|---|---|
| Exp #29 (clean baseline) | None | 0.9988 | 0.9984 | 0.9732 |
| Exp #31 (wm only) | wm=0.1 | 0.9988 | 0.9986 | 0.9681 |
| **Exp #33 (wm + sem)** | wm=0.1, sem=0.005 (gate=0.3) | **0.9988** | **0.9985** | **0.9592** |

On LavaCrossing (only 4 active non-agent object types) the semantic signal is simple — the key question is whether it *hurts* (competing gradient, like reconstruction loss in e2erecon) or is neutral/positive. If neutral, the real test is exp #34 on DoorKey where door/key distinction is semantically critical and hard to learn from sparse reward alone.

**Results (run 1 — buggy):** Peak **0.1990**, Final **0.0000**, Overall avg **0.0129**. **Catastrophic failure** caused by a **tensor layout bug**: `SemanticHead.forward()` reshaped the flat quantized embeddings as `view(B, n_latent, embedding_dim)` but the VQVAE quantized output is `(B, C, H, W)` → channels-first when flattened. The correct reshape is `view(B, embedding_dim, n_latent).permute(0, 2, 1)`. The buggy version fed each position a scramble of mixed channels and spatial locations, producing a random gradient that destabilised the encoder. Not a coefficient or competing-gradient issue — purely a bug.

**Results (run 2 — layout fix, PID 408929):** Peak **0.3927**, Final **0.0000**, Overall avg **N/A**. Still collapsed. Layout bug fixed but `sem_aux_coef=0.05` still too strong — semantic CE ≈ log(11)=2.4 × 0.05 = 0.12 dominates total loss while PPO losses start near zero. Policy never stabilised, reward collapsed to zero by end.

**Results (run 3 — coef=0.005, gate=0.3, PID 1012373):** Peak **0.9988**, Final **0.9985**, Overall avg **0.9592**. **SUCCESS.** Two fixes resolved the failure: (1) `sem_aux_coef` reduced 10× to 0.005 (semantic loss contribution ~0.012, well below PPO signal), (2) reward gate `sem_aux_start_reward=0.3` delays activation until policy is stable. Gate fired at reward=0.3945 (batch ~150). Peak matched exp #31's 0.9988, final (0.9985) and overall (0.9592) slightly below exp #31 (0.9986, 0.9681) but within noise. Semantic aux is **neutral to slightly beneficial** on LavaCrossing — the real test is DoorKey where semantic distinction (door/key) matters.

**Semantic probe comparison (exp31 vs exp33, 30k random frames, linear probe):**

| Class | Exp 31 (wm-only) | Exp 33 (sem-aux) | Improvement |
|-------|-----------------|-----------------|-------------|
| empty | 95.3% | 89.2% | −6.1% |
| wall | 20.3% | **92.7%** | **+72.4%** |
| lava | 0.0% | **46.1%** | **+46.1%** |
| goal | 0.0% | 0.0% | — |
| agent | 0.2% | 1.1% | +0.9% |
| **OVERALL** | 56.2% | **85.2%** | **+29pp** |

The semantic aux dramatically improved wall (20→93%) and partially improved lava (0→46%), confirming the encoder is semantically structured. Failures on goal/agent are due to extreme class imbalance (1.2% each) with unweighted cross-entropy.

---

### Experiment 34: mf_e2e_semantic_aux_v2

**Status:** Pending

**Script:** `run_mf_e2e_semantic_aux_v2.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** Exp 33 achieved 85.2% semantic probe accuracy but failed on rare classes (lava 46%, goal/agent 0%) due to unweighted cross-entropy being dominated by empty (51%) and wall (40%). Adding inverse-frequency class weights (capped at 20×) upweights rare-but-critical classes, a modest coefficient increase (0.005→0.01) strengthens the signal, and an earlier gate (0.3→0.1) gives more training budget.

**Key changes from exp #33:**
1. **Inverse-frequency class weights** in semantic CE loss (capped at 20× to prevent gradient explosion on ultra-rare classes). Expected weights: empty ~1.0, wall ~1.3, lava ~6.8, goal ~20×, agent ~20×.
2. **`sem_aux_coef` 0.005 → 0.01** (2× stronger; total sem contribution ≈ 0.024 at init, still < PPO)
3. **`sem_aux_start_reward` 0.3 → 0.1** (earlier activation; ~100 more batches of semantic training)
4. **Checkpoint now saves** `sem_head_state_dict` and `trans_model_state_dict` (bug fix)

| Parameter | Value |
|---|---|
| Environment | MiniGrid-LavaCrossingS9N1-v0 |
| Encoder | Pretrained VQVAE (hash ea136dc75d389f7b850959cd1f78eb6a) |
| Architecture | filter_size=9, codebook_size=64, embedding_dim=64 (81 tokens) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `e2e_loss` | enabled |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.6 |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| **`sem_aux_coef`** | **0.01** |
| **`sem_aux_start_reward`** | **0.1** |
| **`sem_class_weights`** | **enabled (inverse-freq, cap=20×)** |

**Comparison baseline:**

| Experiment | Aux losses | Peak | Final | Overall avg | Probe % |
|---|---|---|---|---|---|
| Exp #31 (wm only) | wm=0.1 | 0.9988 | 0.9986 | 0.9681 | 56.2% |
| Exp #33 (wm + sem) | wm=0.1, sem=0.005 | 0.9988 | 0.9985 | 0.9592 | 85.2% |
| **Exp #34 (wm + sem v2)** | wm=0.1, sem=0.01 + weights | **0.9987** | **0.9984** | **0.9526** | **84.4%** |

**Results:** Peak **0.9987**, Final **0.9984**, Overall avg **0.9526**. Gate fired at reward=0.1786. Class weights computed: empty=0.2, wall=0.2, goal=7.4, lava=1.2, agent=7.4. RL performance matches exp 33 — semantic aux with class weights does not hurt. However, **semantic probe did NOT improve as expected**: 84.4% vs exp 33's 86.0%, with lava slightly worse (45.2% vs 49.7%), goal still 0%, agent still 0%. The class weights correctly upweighted rare classes (goal/agent at 7.4×) but the encoder still lacks the capacity or gradient signal to distinguish these ultra-rare objects. Wall improved slightly (94.9% vs 92.8%). The issue is likely that lava/goal/agent occupy too few spatial positions for the encoder CNN to develop dedicated features for them — the ~7× lava upweight was insufficient against the 50%+40% gradient mass of empty+wall.

---

### Experiment 35: mf_e2e_semantic_doorkey

**Status:** Done

**Script:** `run_mf_e2e_semantic_doorkey.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** DoorKey-8x8 is the real test for semantic grounding. Unlike LavaCrossing (5 classes, lava is the only hazard), DoorKey requires distinguishing **door**, **key**, and **goal** — objects the agent must interact with in sequence (find key → open door → reach goal). A stronger SemanticHeadV2 with architectural improvements should produce measurably better semantic representations and potentially improve RL sample efficiency.

**Architecture: SemanticHeadV2** (new):
```
embed (64-dim) + pos_embed (64-dim, learnable) → concat (128-dim)
    → 3×3 depth-wise conv on 9×9 grid (local context)
    → LayerNorm
    → Linear(128→128) → ReLU → LayerNorm → Linear(128→128) → ReLU → Linear(128→11)
```
- **Positional encoding**: learnable 64-dim vector per grid position (81 positions). Encodes spatial priors (walls at edges, objects in interior).
- **Local context**: 3×3 depth-wise convolution lets each position see its 8 neighbors, breaking pure per-position independence. Helps detect boundaries (wall-to-empty transitions) and small objects (key = 1 position).
- **Deeper MLP**: 3 layers with LayerNorm (vs 2 layers in v1). More capacity for multi-class separation.
- **Focal loss** (γ=2): Down-weights easy examples (wall, empty) exponentially, focusing gradient on hard cases (door, key, goal, agent).

**Key changes from exp #32 (DoorKey wm-only baseline):**
- `--use_semantic_aux` with SemanticHeadV2
- `--sem_aux_coef 0.05` (aggressive — 5× exp 34, proven safe on LavaCrossing)
- `--sem_focal_gamma 2.0` (focal loss)
- `--sem_class_weights` (inverse-frequency, cap 20×)
- `--sem_head_version 2` (new architecture)
- `--sem_aux_start_reward 0.1` (early gate for DoorKey's slower learning)

| Parameter | Value |
|---|---|
| Environment | **MiniGrid-DoorKey-8x8-v0** |
| Encoder | **Random init** (no pretrained VQVAE) |
| Architecture | filter_size=9, codebook_size=64, embedding_dim=64 (81 tokens) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.1 |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| **`sem_aux_coef`** | **0.05** |
| **`sem_head_version`** | **2 (SemanticHeadV2)** |
| **`sem_head_hidden`** | **128** |
| **`sem_focal_gamma`** | **2.0** |
| **`sem_class_weights`** | **enabled** |
| **`sem_aux_start_reward`** | **0.1** |

**DoorKey active classes:** empty(1), wall(2), floor(3), door(4), key(5), goal(8), agent(10) — 7 of 11 classes active (vs 5 in LavaCrossing).

**Comparison baseline:**

| Experiment | Environment | Aux losses | Peak | Final | Overall |
|---|---|---|---|---|---|
| Exp #32 (DoorKey wm-only) | DoorKey-8x8 | wm=0.1 | 0.9988 | 0.9979 | 0.9979 |
| **Exp #35 (DoorKey sem v2)** | DoorKey-8x8 | wm=0.1, sem=0.05 + focal | **0.9988** | **0.9943** | **0.9974** |

**Results:**

- **RL Performance:** Peak 0.9988, Final 0.9943, Overall avg 0.9974 — matches exp 32 baseline (0.9979). Semantic aux did not hurt RL.
- **Semantic Probe (linear, 30k frames):** **58.4% overall** — WORSE than exp 32 baseline (62.4%).

| Class | Exp 32 (wm-only) | Exp 35 (sem v2) |
|---|---|---|
| empty | 67.2% | 64.6% |
| wall | 76.3% | 61.0% |
| floor | 0.0% | 0.0% |
| door | 0.0% | 0.2% |
| key | 0.0% | 0.0% |
| goal | 0.0% | 0.0% |
| agent | 0.0% | 0.0% |

**Analysis — DISAPPOINTING:**
1. SemanticHeadV2 (pos encoding + local conv + focal loss + class weights) did NOT improve semantic grounding despite 10× more parameters (41k vs 5k).
2. Wall accuracy actually **regressed** (76% → 61%), suggesting the aggressive coef=0.05 + focal loss may have destabilized encoder gradients for majority classes.
3. Rare objects (door 0.2%, key/goal/agent 0%) remain invisible to the linear probe — same failure as LavaCrossing experiments.
4. **Root cause hypothesis:** The bottleneck is the VQVAE encoder architecture, not the semantic head. The CNN encoder uses `AdaptiveAvgPool2d(9×9)` which spatially blurs features, making it inherently hard to preserve fine-grained per-position object identity. Small objects (key=1 cell, agent=1 cell) get averaged into their surroundings. No amount of classification head improvement can recover information lost at the encoder level.

---

### Experiment 36: mf_e2e_semantic_doorkey_v5enc

**Status:** Done

**Script:** `run_mf_e2e_semantic_doorkey_v5enc.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** The v2 encoder's `AdaptiveAvgPool2d` is the bottleneck for semantic grounding. It **upsamples** from 5×5 → 9×9, creating blurry interpolated features where spatial boundaries between grid cells are destroyed. A strided convolution encoder (v5) that naturally produces 8×8 output from 64×64 input — with each token mapping to exactly one DoorKey grid cell — should dramatically improve semantic probe accuracy.

**Architecture: Encoder v5** (new):
```
Input (3, 64, 64)
  → Conv2d(3→64, k=4, s=2, p=1) + ReLU → (64, 32, 32)    # 2× downsample
  → Conv2d(64→128, k=4, s=2, p=1) + ReLU → (128, 16, 16)  # 2× downsample
  → Conv2d(128→64, k=4, s=2, p=1) + ReLU → (64, 8, 8)     # 2× downsample
                                                              # 64 tokens, NO pooling
```

**Why this should work:**
1. **No adaptive pooling:** v2 upsampled 5×5 → 9×9 via `AdaptiveAvgPool2d`, averaging features across cell boundaries. v5 never pools — each token's receptive field is naturally bounded.
2. **1:1 grid alignment:** DoorKey-8x8 has 8×8 grid, rendered at 8 pixels/tile = 64×64 image. Three stride-2 layers produce exactly 8×8 = 64 tokens. Each token corresponds to exactly one grid cell. No resizing of semantic labels needed.
3. **Smaller receptive field:** v5 RF = 22×22 pixels (≈2.75 cells). v2 RF = 30 pixels + global pooling. The tighter RF means each token encodes more local information.
4. **Fewer parameters:** v5 encoder has 265k params vs v2's 439k. Less capacity means less room to learn shortcuts that ignore spatial structure.

**Key changes from exp #35:**

| Setting | Exp 35 (v2 encoder) | Exp 36 (v5 encoder) |
|---|---|---|
| `ae_model_version` | 2 | **5** |
| `filter_size` | 9 (forced via AdaptiveAvgPool) | **8** (natural stride output) |
| Encoder output | (64, 9, 9) = 81 tokens | **(64, 8, 8) = 64 tokens** |
| Downsampling | Conv→5×5, then pool↑9×9 | **Conv→8×8 directly** |
| Grid alignment | Semantic labels resized 8→9 | **No resize needed (8=8)** |
| Encoder params | 438,528 | **265,472** |
| All other params | identical | identical |

| Parameter | Value |
|---|---|
| Environment | **MiniGrid-DoorKey-8x8-v0** |
| Encoder | **v5 strided** (random init) |
| Architecture | filter_size=8, codebook_size=64, embedding_dim=64 (**64 tokens**) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.1 |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `sem_aux_coef` | 0.05 |
| `sem_head_version` | 2 (SemanticHeadV2) |
| `sem_head_hidden` | 128 |
| `sem_focal_gamma` | 2.0 |
| `sem_class_weights` | enabled |
| `sem_aux_start_reward` | 0.1 |

**Comparison baseline:**

| Experiment | Encoder | Tokens | Peak | Final | Overall | Probe |
|---|---|---|---|---|---|---|
| Exp #32 (DoorKey wm-only) | v2 + pool | 81 (9×9) | 0.9988 | 0.9979 | 0.9979 | 62.4% |
| Exp #35 (DoorKey sem v2) | v2 + pool | 81 (9×9) | 0.9988 | 0.9943 | 0.9974 | 58.4% |
| **Exp #36 (DoorKey v5 enc)** | **v5 strided** | **64 (8×8)** | **0.9988** | **0.9984** | TBD | **82.9%** |

**Results:**

- **RL Performance:** Peak 0.9988, Final 0.9984 — matches all baselines. Encoder change did not hurt RL.
- **Semantic Probe (linear, 30k frames):** **82.9% overall** — MASSIVE improvement over exp 32 (63.6%) and exp 35 (58.3%).

| Class | Exp 32 (v2, wm-only) | Exp 35 (v2, sem v2) | **Exp 36 (v5, sem v2)** |
|---|---|---|---|
| empty | 59.1% | 67.3% | **67.0%** |
| wall | 76.3% | 56.8% | **97.3%** ↑21 |
| door | 0.0% | 1.8% | **94.4%** ↑94 |
| key | 0.0% | 0.0% | **100.0%** ↑100 |
| goal | 0.0% | 0.0% | **0.0%** (unchanged) |
| agent | 0.0% | 0.0% | **99.8%** ↑100 |

**Analysis — BREAKTHROUGH:**
1. **Hypothesis confirmed:** The bottleneck was the encoder, not the semantic head. Replacing `AdaptiveAvgPool2d` with strided convolutions unlocked dramatic semantic grounding improvements.
2. **1:1 grid alignment is critical.** The v5 encoder produces 8×8 tokens from a 64×64 image — each token covers exactly one 8×8 pixel tile = one MiniGrid grid cell. No resize, no interpolation, no blurring.
3. **Key/door/agent went from invisible (0%) to near-perfect (94–100%).** These small single-cell objects were previously averaged away by the pooling layer.
4. **Wall accuracy jumped from 76% → 97%.** Precise spatial boundaries mean each border position cleanly maps to "wall" vs "empty".
5. **Goal remains at 0%.** This is the one remaining failure. Hypothesis: the goal tile's visual appearance (green square) may be too similar to empty (black) after the strided convolutions; or goal position is confounded with empty because both appear in the interior.
6. **Fewer parameters worked better:** v5 encoder has 266k params vs v2's 439k. The simpler architecture with clean spatial structure outperforms the larger one with pooling artifacts.

---

### Experiment 37: mf_e2e_semantic_doorkey_v5enc_prevq

**Status:** Done

**Script:** `run_mf_e2e_semantic_doorkey_v5enc_prevq.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** Exp 36 achieved 82.9% probe accuracy but goal remained at 0%. Codebook analysis revealed that 99.8% of goal samples map to code #4 (wall's dominant code) — and even pre-VQ continuous features show goal=0%. The encoder never learned to separate goal because STE gradients through VQ are too noisy for a 1.6%-prevalence class. Applying semantic loss directly to pre-VQ encoder output gives clean gradient without the STE bottleneck, which should force the encoder to learn goal-discriminative features. Larger codebook (256 vs 64) then gives goal its own code.

**Codebook diagnosis (exp 36):**
```
goal  → code #4 (99.8%)     ← SAME as wall's dominant code (81%)!
key   → codes 20,24,35      ← 0% overlap with empty → 100% probe accuracy
agent → codes 31,3,7,32,50  ← 0% overlap with empty → 99.8% probe accuracy
```
**Root cause:** VQ codebook has 64 entries. Goal's encoder features are close enough to wall that they snap to the same code. Even pre-VQ continuous features don't separate them (goal probe=0% pre-VQ too), confirming the encoder itself is the bottleneck, not quantization.

**Key changes from exp #36:**

| Setting | Exp 36 (post-VQ sem) | Exp 37 (pre-VQ sem) |
|---|---|---|
| `sem_pre_vq` | False | **True** |
| `codebook_size` | 64 | **256** |
| `sem_focal_gamma` | 2.0 | **0.0** (standard CE) |
| All other params | (identical) | (identical) |

**Why pre-VQ semantic loss should help:**
1. **Direct gradient:** The semantic head gets continuous encoder features, not quantized ones. Backprop flows straight to the encoder without STE approximation.
2. **Clean signal for rare classes:** STE gradient for a 1.6%-prevalence class like goal gets drowned by the 98% majority-class signal. Direct gradient preserves the full loss signal.
3. **Larger codebook (256):** Once the encoder learns to separate goal, 256 entries (vs 64) provide enough capacity for goal to claim its own codebook vector.
4. **No focal loss:** γ=2 focal may over-focus on already-hard-to-classify positions, starving the rare-but-distinctive classes (goal). Standard CE with class weights is more balanced.

| Parameter | Value |
|---|---|
| Environment | **MiniGrid-DoorKey-8x8-v0** |
| Encoder | **v5 strided** (random init) |
| Architecture | filter_size=8, **codebook_size=256**, embedding_dim=64 (64 tokens) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.1 |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `sem_aux_coef` | 0.05 |
| **`sem_pre_vq`** | **True** |
| `sem_head_version` | 2 (SemanticHeadV2) |
| `sem_head_hidden` | 128 |
| **`sem_focal_gamma`** | **0.0** (standard CE) |
| `sem_class_weights` | enabled |
| `sem_aux_start_reward` | 0.1 |

**Comparison baseline:**

| Experiment | Codebook | Sem Loss | Probe | Goal |
|---|---|---|---|---|
| Exp #32 (v2, wm-only) | 64 | — | 63.6% | 0% |
| Exp #36 (v5, post-VQ sem) | 64 | post-VQ (STE) | 82.9% | 0% |
| **Exp #37 (v5, pre-VQ sem)** | **256** | **pre-VQ (direct)** | **83.5%** | **0.0%** |

**Results:**

- **RL Performance:** Peak 0.9988, Final 0.9982 — matches all baselines.
- **Semantic Probe (linear, 30k frames):** **83.5% overall** — marginal improvement over exp 36 (82.1%).

| Class | Exp 32 (v2, wm-only) | Exp 36 (v5, post-VQ sem) | **Exp 37 (v5, pre-VQ sem)** |
|---|---|---|---|
| empty | 57.9% | 65.4% | **69.0%** ↑ |
| wall | 73.5% | 97.1% | **96.8%** ≈ |
| door | 0.1% | 85.9% | **94.5%** ↑ |
| key | 0.0% | 100.0% | **97.9%** ↓ |
| goal | 0.0% | 0.0% | **0.0%** ✗ |
| agent | 0.0% | 99.8% | **100.0%** ↑ |

**Analysis — GOAL REMAINS AT 0%:**
1. Pre-VQ semantic loss + larger codebook (256) improved overall accuracy slightly (82.1% → 83.5%) and improved empty/door, but **goal is still 0%**.
2. The pre-VQ semantic head re-encodes raw observations through `ae_model.encoder` during PPO updates. This gives direct gradient to the encoder, but the encoder still fails to produce goal-discriminative features.
3. **Possible explanations:**
   - The semantic loss is gated (only activates after reward > 0.1). By the time it activates, the encoder may already be locked into a representation that doesn't separate goal.
   - The encoder learning rate (1e-5 with cosine decay → 0) may be too small to restructure features for a 1.6% class by the time the semantic signal kicks in.
   - Goal's solid green tile may genuinely produce similar low-level CNN features to wall (both are uniform-color regions), requiring deeper architectural changes (e.g., color-channel attention) rather than just loss signal improvements.
4. **Key lesson:** The encoder architecture (v5 strided) was the decisive factor for semantic grounding (exp 36: 62.5% → 82.1%). The loss signal improvements (pre-VQ, larger codebook) provide diminishing returns (82.1% → 83.5%). Goal may require encoder-level changes to resolve.

---

### Experiment 38: mf_e2e_semantic_doorkey_v6enc_goal

**Status:** Complete ✅

**Script:** `run_mf_e2e_semantic_doorkey_v6enc_goal.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** Exp 36 fixed spatial alignment and exp 37 gave the semantic loss a cleaner gradient, but goal still stayed at 0%. That points to a remaining encoder bottleneck: the latent tokens still do not preserve the weak per-cell RGB signal of the rare green goal tile strongly enough before quantization. A new encoder that keeps v5's 8×8 grid alignment while injecting an explicit pooled RGB shortcut and x/y coordinates into each token should make goal easier to separate from empty/wall. Starting semantic supervision immediately should also prevent the encoder from settling into a wall/empty-biased representation before the goal loss turns on.

**Key changes from exp #37:**

| Setting | Exp 37 | Exp 38 |
|---|---|---|
| `ae_model_version` | 5 | **6** |
| Encoder path | strided conv only | **strided conv + pooled RGB + coord channels** |
| `sem_aux_start_reward` | 0.1 | **0.0** |
| `sem_aux_coef` | 0.05 | **0.08** |
| `codebook_size` | 256 | **256** |
| `sem_pre_vq` | True | **True** |

**Architecture: Encoder v6** (new):
```
Input (3, 64, 64)
  → pooled RGB shortcut to (16, 8, 8)
  → x/y coordinate grid projected to (8, 8, 8)
  → conv trunk on [RGB + coords]:
      Conv(5→64, s=2) → Conv(64→128, s=2) → Conv(128→40, s=2)
  → concat(trunk, pooled_rgb, coord_proj) = (64, 8, 8)
  → 1×1 fusion + residual block
```

**Why this should help goal:**
1. **Color is preserved explicitly.** The goal is a rare but distinctive green tile. A pooled-RGB shortcut gives every token direct access to its cell-local color statistics instead of requiring the conv trunk to preserve them implicitly.
2. **Coordinates reduce ambiguity.** Empty interior cells and the goal can share simple local texture. Adding x/y channels gives the encoder a light positional scaffold without destroying the 1:1 token-to-cell mapping.
3. **Early semantic shaping.** In exp 37, semantic loss only activated after reward > 0.1. With a very small encoder LR, that may be too late for the rare goal class to carve out its own feature direction.

| Parameter | Value |
|---|---|
| Environment | **MiniGrid-DoorKey-8x8-v0** |
| Encoder | **v6 color+coord** (random init) |
| Architecture | filter_size=8, codebook_size=256, embedding_dim=64 (**64 tokens**) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.1 |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `sem_aux_coef` | **0.08** |
| `sem_head_version` | 2 (SemanticHeadV2) |
| `sem_pre_vq` | **True** |
| `sem_focal_gamma` | 0.0 |
| `sem_class_weights` | enabled |
| `sem_aux_start_reward` | **0.0** |

**Success criterion:**
- Keep RL performance at or near exp 36/37 (`peak ≈ 0.9988`)
- Improve semantic probe beyond exp 37's `83.5%`
- Most importantly: raise **goal** above `0%`

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.9528** |
| Final 10-ep avg | 0.0898 (end-of-training LR→0 collapse) |
| Semantic probe overall | **91.1%** (best across all experiments) |
| Semantic probe goal | **🟢 98.8%** (was 0% in all prior experiments) |

**Per-class probe accuracy:**

| Class | Exp 36 (v5) | Exp 37 (v5+preVQ) | Exp 38 (v6) |
|-------|-------------|-------------------|-------------|
| empty | 67.2% | 54.7% | **89.6%** |
| wall | 97.1% | 97.1% | 93.0% |
| door | 92.2% | 98.3% | 59.4% ⬇ |
| key | 100.0% | 96.7% | 81.8% ⬇ |
| **goal** | 0.0% | 0.0% | **98.8%** 🟢 |
| agent | 99.9% | 100.0% | 98.0% |
| **OVERALL** | 82.9% | 77.7% | **91.1%** |

**Analysis:**
- Goal grounding fully solved by the pooled RGB shortcut: each token receives direct per-cell average color, making the solid green goal tile unambiguous to the encoder even before quantization.
- Empty accuracy also jumped significantly (67% → 90%), as explicit RGB and coordinate features help differentiate interior empty cells from walls.
- Door and key accuracy regressed (door: 98% → 59%, key: 97% → 82%). These are object-level features that require fine-grained local texture; the pooled RGB shortcut averages over the cell and may dilute those signals. The conv trunk is also narrower in v6 (trunk_dim = 40 vs 64 in v5).
- End-of-training reward collapse is a known issue with cosine LR → 0; best model (0.9528) is still competitive with exp 36/37.
- **Next step:** Exp 39 — restore door/key accuracy by widening the trunk or using a multi-scale color shortcut (3×3 patch pooling in addition to full-cell pooling).

---

### Experiment 39: mf_e2e_semantic_doorkey_v7enc_multiscale

**Status:** Pending

**Script:** `run_mf_e2e_semantic_doorkey_v7enc_multiscale.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** Exp 38's v6 encoder solved goal (0% → 98.8%) but regressed on door (98% → 59%) and key (100% → 82%). The v6 approach was MiniGrid-specific (hand-crafted RGB pooling at exact tile size, hard-coded coordinate grids, narrowed trunk). A generalised multi-scale architecture should recover door/key while retaining goal, by letting the network learn *which* features matter rather than prescribing them.

**Key changes from exp #38:**

| Setting | Exp 38 (v6) | Exp 39 (v7) |
|---|---|---|
| `ae_model_version` | 6 | **7** |
| Encoder design | RGB shortcut + coord grid (MiniGrid-specific) | **Multi-scale skips + SE attention + learned pos embed (general)** |
| Trunk width | 40 channels (narrowed for RGB/coord paths) | **64 channels (full width)** |
| Skip mechanism | AvgPool2d at exact tile size (8×8) | **Adaptive pool from intermediate conv layers** |
| Position encoding | Hard-coded x/y coordinate grid | **Learnable spatial embeddings** |
| Channel selection | Fixed colour/coord/trunk split | **Squeeze-and-Excitation learned attention** |

**Architecture: Encoder v7** (new):
```
Input (C, 64, 64)
  → L1: Conv(C→64,  k=4, s=2, p=1) + ReLU   → f1 (64, 32, 32)
  → L2: Conv(64→128, k=4, s=2, p=1) + ReLU   → f2 (128, 16, 16)
  → L3: Conv(128→64, k=4, s=2, p=1) + ReLU   → f3 (64, 8, 8)

  Skip-1: AdaptiveAvgPool(f1 → 8×8) → Conv1×1(64→16) → s1 (16, 8, 8)
  Skip-2: AdaptiveAvgPool(f2 → 8×8) → Conv1×1(128→16) → s2 (16, 8, 8)

  cat([f3, s1, s2]) = (96, 8, 8)
  → Conv1×1(96→64) + ReLU → (64, 8, 8)
  → SE channel attention (reduction=4)
  → + learnable pos_embed (1, 64, 8, 8)
  → ResidualBlock → output (64, 8, 8)
```

**Why this should improve on v6:**
1. **Multi-scale skips preserve texture.** Skip-1 has 4×4 receptive field — fine enough to capture door's keyhole pattern and key's distinctive shape. v6's AvgPool averaged over the full 8×8 tile, destroying these details.
2. **SE attention is adaptive.** Instead of fixed 40/16/8 channel allocation, SE learns per-sample weights: upweight colour channels for uniform regions (goal/wall), upweight texture channels for object regions (door/key).
3. **Full-width backbone.** 64 channels in the last conv (vs 40 in v6) gives more capacity for all classes.
4. **Learned positions generalise.** Hard-coded coordinates assume grid-world structure; learnable positional embeddings can capture whatever spatial patterns the data exhibits.

| Parameter | Value |
|---|---|
| Environment | **MiniGrid-DoorKey-8x8-v0** |
| Encoder | **v7 multi-scale + SE** (random init) |
| Architecture | filter_size=8, codebook_size=256, embedding_dim=64 (**64 tokens**) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.1 |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `sem_aux_coef` | 0.08 |
| `sem_head_version` | 2 (SemanticHeadV2) |
| `sem_pre_vq` | True |
| `sem_focal_gamma` | 0.0 |
| `sem_class_weights` | enabled |
| `sem_aux_start_reward` | 0.0 |

**Success criterion:**
- Retain goal probe accuracy above 90% (was 98.8% in exp 38)
- Recover door above 80% (was 59.4% in exp 38, 98.3% in exp 37)
- Recover key above 90% (was 81.8% in exp 38, 100% in exp 36)
- Overall probe above 91% (exp 38 level)
- RL reward competitive with exp 38 (best ≈ 0.95)

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | — |
| Final 10-ep avg | — |
| Semantic probe overall | — |
| Semantic probe goal | — |
| Semantic probe door | — |
| Semantic probe key | — |

---

### Experiment 40: mf_e2e_semantic_doorkey_v8enc_gated

**Status:** Complete ✅

**Script:** `run_mf_e2e_semantic_doorkey_v8enc_gated.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** Across experiments, v5's full-width trunk (64ch) is consistently best for structural/textural classes (wall 96.5%, door 99.4%, key 100%), while v6's input-level shortcut was the key to cracking goal (0% → 98%). v7's attempt to generalise with SE attention, multi-scale skips, and learned positions actually hurt wall badly (96% → 78%) and failed to recover door. The optimal design is a **minimal intervention on v5**: keep the proven trunk exactly, add only a lightweight gated input skip that the network can selectively enable per-token, and avoid the components that hurt in v7.

**Key changes from exp #39 (v7):**

| Setting | Exp 39 (v7) | Exp 40 (v8) |
|---|---|---|
| `ae_model_version` | 7 | **8** |
| Trunk width | 64ch (restored from v6's 40) | **64ch (identical to v5)** |
| Skip type | Multi-scale from L1 + L2 | **Input-level only (pooled raw obs)** |
| Channel attention | SE block (reduction=4) | **None (removed — hurt wall in v7)** |
| Position encoding | Learnable spatial embed | **None (removed — hurt wall in v7)** |
| Fusion | Concatenate + 1×1 conv | **Learned sigmoid gate (residual)** |

**Architecture: Encoder v8** (new):
```
Input (C, 64, 64)
  ├─ Trunk (v5-identical):
  │    Conv(C→64, k=4, s=2) → Conv(64→128, k=4, s=2) → Conv(128→64, k=4, s=2)
  │    → f_trunk (64, 8, 8)
  │
  └─ Input skip:
       AdaptiveAvgPool2d(8) → Conv1×1(C→16) → ReLU → Conv1×1(16→16) → ReLU
       → f_skip (16, 8, 8)

  cat([f_trunk, f_skip]) = (80, 8, 8)
    → gate  = σ(Conv1×1(80→64))                ∈ [0, 1]  per-token per-channel
    → merge = ReLU(Conv1×1(80→64))
    → out   = gate · merge + (1 − gate) · f_trunk     (gated residual)
    → ResidualBlock → final (64, 8, 8)
```

**Why this should beat v7:**
1. **v5 trunk preserved exactly.** Wall/door accuracy depends on the full 64-ch trunk — v8 keeps it untouched.
2. **Gated residual is safe.** When gate ≈ 0, the output is pure v5. The skip can only add information, never override. This avoids the v7 failure mode where SE + positions corrupted wall features.
3. **Input skip is general.** Pools raw obs at the latent resolution — captures colour, texture, whatever the input contains. Works for any visual domain, not just grid worlds.
4. **Minimal new parameters.** Only 85K extra params (v5: 265K → v8: 350K). No complex multi-branch architecture.

| Parameter | Value |
|---|---|
| Environment | **MiniGrid-DoorKey-8x8-v0** |
| Encoder | **v8 gated input skip** (random init) |
| Encoder params | 350,160 (v5: 265,472) |
| Architecture | filter_size=8, codebook_size=256, embedding_dim=64 (**64 tokens**) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.1 |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `sem_aux_coef` | 0.08 |
| `sem_head_version` | 2 (SemanticHeadV2) |
| `sem_pre_vq` | True |
| `sem_focal_gamma` | 0.0 |
| `sem_class_weights` | enabled |
| `sem_aux_start_reward` | 0.0 |

**Success criterion:**
- Wall above 90% (v5: 96.5%, v7: 78.5%)
- Door above 85% (v5: 99.4%, v7: 71.9%)
- Key above 95% (v5: 100%, v7: 99.1%)
- Goal above 90% (v6: 98.1%, v7: 94.0%)
- Overall probe above 85%
- RL reward ≥ 0.95

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.9988** |
| Final 10-ep avg | **0.9985** (no collapse) |
| Semantic probe overall | **71.0%** ⬇ |
| Semantic probe goal | **95.0%** ✅ |
| Semantic probe door | 77.9% |
| Semantic probe key | **98.4%** ✅ |
| Semantic probe wall | 83.3% |
| Semantic probe empty | 53.1% ⬇ |
| Semantic probe agent | 92.7% |

**Analysis:**
- RL reward is perfect (0.9988 best, no end-of-training collapse).
- Goal retained at 95% — the gated input skip works for color-identity preservation.
- Key excellent (98.4%) — trunk capacity preserved.
- **However:** empty collapsed to 53% (worst ever) and wall stayed at 83% (same as v7, far from v5's 96%). Overall 71% is the worst of all encoder variants.
- **Diagnosis:** The gated fusion changes gradient flow through the trunk even when the gate is near 0. The sigmoid gate + merge MLP create an additional optimization path that the encoder exploits, shifting what the trunk learns. The v5 trunk architecture is preserved but its *learned features* are different because the loss landscape changed.
- **Key lesson:** Modifications to v5 (shortcuts, gates, attention) consistently degrade wall/empty regardless of design. The problem may not be solvable by adding components to a strided-conv encoder — a fundamentally different encoding strategy is needed.

---

### Experiment 41: mf_e2e_semantic_doorkey_v9enc_patch

**Status:** Complete ✅

**Script:** `run_mf_e2e_semantic_doorkey_v9enc_patch.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** All experiments v5–v8 use overlapping strided convolutions whose receptive fields span multiple tiles. Every modification to help rare classes (goal) degrades dominant classes (wall/empty) because the changes alter gradient flow through the shared trunk. A fundamentally different architecture is needed: **ViT-style patch embedding** that processes each tile independently through a per-patch MLP, then adds cross-token context as a residual. This guarantees per-tile identity preservation by construction, not via shortcuts.

**Key architectural difference from ALL prior encoders:**

| Property | v5–v8 (strided conv) | v9 (patch embed) |
|---|---|---|
| Patch processing | Overlapping kernels mix cross-tile info | **Non-overlapping: each tile processed independently** |
| Receptive field | Grows with depth (12–22 pixels across tiles) | **Exactly 1 tile per token in Phase 1** |
| Cross-tile context | Built into the conv trunk from layer 1 | **Added as residual AFTER independent embedding** |
| Color preservation | Requires shortcuts/gates (fragile) | **Inherent — raw pixels enter MLP directly** |
| Architecture family | CNN | **ViT patch projection + conv context** |

**Architecture: Encoder v9** (new):
```
Phase 1 — Independent patch embedding:
  Input (3, 64, 64)
    → Conv2d(3, 256, k=8, s=8)    [non-overlapping = per-patch linear]
    → GroupNorm(32) → GELU
    → Conv2d(256, 64, k=1)         [project to embedding dim]
    → GELU
    → (64, 8, 8)   each token = one tile, processed independently

Phase 2 — Local context refinement (residual):
    → Conv2d(64, 64, k=3, p=1) → GroupNorm(16) → GELU
    → Conv2d(64, 64, k=3, p=1) → GroupNorm(16)
    → add residual from Phase 1
    → GELU
    → (64, 8, 8)
```

**Why this should work where v5–v8 couldn't:**
1. **No cross-tile contamination in Phase 1.** Goal's green pixels and wall's grey pixels are processed by completely separate MLP evaluations. Their embeddings are maximally different by construction.
2. **Context is additive only.** Phase 2 is a residual — it can enrich the per-tile identity but cannot erase it. This is structurally different from v8's gated fusion which changed gradient flow through the trunk.
3. **Lightweight.** Only 140K encoder params (v5: 265K, v8: 350K). Fewer parameters = less capacity for the optimizer to "waste" on task-irrelevant features.
4. **Truly domain-agnostic.** Standard ViT patch projection — no RGB shortcuts, coordinates, SE blocks, or gating. Works for any visual input.

| Parameter | Value |
|---|---|
| Environment | **MiniGrid-DoorKey-8x8-v0** |
| Encoder | **v9 patch + context** (random init) |
| Encoder params | **140,480** (lightest encoder) |
| Architecture | filter_size=8, codebook_size=256, embedding_dim=64 (**64 tokens**) |
| `mf_steps` | 5,000,000 |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 (policy/critic) |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=0.1 |
| `use_world_model` | enabled |
| `wm_aux_coef` | 0.1 |
| `sem_aux_coef` | 0.08 |
| `sem_head_version` | 2 (SemanticHeadV2) |
| `sem_pre_vq` | True |
| `sem_focal_gamma` | 0.0 |
| `sem_class_weights` | enabled |
| `sem_aux_start_reward` | 0.0 |

**Success criterion:**
- Goal above 90% (inherent from patch independence)
- Wall above 90% (no cross-tile contamination to degrade it)
- Door above 80%
- Key above 95%
- Overall above 85%
- RL reward ≥ 0.95

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | **0.9988** |
| Final 10-ep avg | **0.9982** (no collapse) |
| Semantic probe overall | **78.0%** |
| Semantic probe goal | **0.0%** ⬇ (back to zero!) |
| Semantic probe door | 75.0% |
| Semantic probe key | **100%** ✅ |
| Semantic probe wall | 82.9% |
| Semantic probe empty | 73.3% |
| Semantic probe agent | **99.3%** |

**Analysis — CRITICAL FINDING:**
Goal returned to 0% despite non-overlapping patch processing. This **disproves the receptive-field contamination hypothesis** — the problem is NOT that neighboring tiles corrupt the goal token's embedding.

The real bottleneck is the **VQ codebook**: with EMA updates dominated by wall (52%) and empty (42%), goal's 1.6% of tokens never claims a dedicated code. The encoder may produce distinct embeddings for goal, but the VQ layer maps them to the same codes as other classes.

**Only v6/v7/v8 solved goal** — all of which had explicit mechanisms (RGB shortcut, multi-scale skips, gated input skip) that made the *pre-VQ* continuous embedding for goal geometrically distant enough from wall/empty that the codebook was forced to allocate distinct codes. The patch encoder produces different features for goal, but not different *enough* to survive VQ compression.

**Implication:** The fix must happen at the VQ level (dead-code restart, codebook diversity pressure) or the encoder must produce maximally separated pre-VQ features (which v6's color path achieved). Architecture alone isn't sufficient without either VQ reform or strong inductive bias.

---

### Experiment 42: mf_e2e_semantic_doorkey16_v6enc_restart

**Status:** Pending

**Script:** `run_mf_e2e_semantic_doorkey16_v6enc_restart.sh` → `discrete_mbrl/model_free/train.py`

**Hypothesis:** v6 proved best overall (88.9%) on DoorKey-8x8 and was the only architecture to solve all six classes including goal (97.8%). This experiment scales to a harder environment (DoorKey-16×16) while adding VQ dead-code restart to address the codebook collapse that prevents rare classes from claiming codes. The combination of v6's color-aware encoder + codebook restart should maintain semantic grounding even as rare-class frequency drops from 1.6% to 0.4%.

**Environment change: DoorKey-8x8 → DoorKey-16x16:**

| Property | DoorKey-8x8 | DoorKey-16x16 |
|---|---|---|
| Grid | 8×8 | **16×16** |
| Image size | (3, 64, 64) | **(3, 128, 128)** |
| Tokens | 64 | **256** |
| Goal frequency | 1.6% | **0.4%** |
| Wall frequency | 51.6% | **28.5%** |
| Empty frequency | 42.4% | **69.9%** |
| Exploration difficulty | Easy | **Hard** (much larger rooms) |

**Key changes from exp #38 (v6 on 8x8):**

| Setting | Exp 38 (8x8) | Exp 42 (16x16) |
|---|---|---|
| Environment | DoorKey-8x8 | **DoorKey-16x16** |
| Image size | (3, 64, 64) | **(3, 128, 128)** |
| `filter_size` | 8 | **16** (1:1 cell alignment) |
| Tokens | 64 | **256** |
| `codebook_size` | 256 | **512** (more diversity for 256 tokens) |
| `mf_steps` | 5M | **8M** (harder task) |
| VQ dead-code restart | None | **threshold=1.0** (resets dead codes from encoder samples) |
| `snapback_min_reward` | 0.1 | **0.05** (harder task, lower initial reward) |

**Architecture: Encoder v6 on 128×128:**
```
Input (3, 128, 128)
  → pooled RGB: AvgPool2d(8,8) → (3, 16, 16) → Conv1×1(3→16) → (16, 16, 16)
  → coordinates: meshgrid(16×16) → Conv1×1(2→8) → (8, 16, 16)
  → trunk: [RGB+coords] → Conv(5→64, s=2) → Conv(64→128, s=2) → Conv(128→40, s=2) → (40, 16, 16)
  → cat(trunk, color, coord) = (64, 16, 16)
  → 1×1 fusion + residual → (64, 16, 16) = 256 tokens
```

**VQ dead-code restart mechanism:**
```
At each EMA update step:
  1. Track cluster_usage via EMA (existing)
  2. Find codes where cluster_usage < 1.0 (dead)
  3. Reinitialize dead codes with random encoder outputs from current batch
  4. Reset their EMA statistics to give them a fair start
```

| Parameter | Value |
|---|---|
| Environment | **MiniGrid-DoorKey-16x16-v0** |
| Encoder | **v6 color+coord** |
| Architecture | filter_size=16, codebook_size=**512**, embedding_dim=64 (**256 tokens**) |
| State dim | 256 × 64 = **16,384** |
| `mf_steps` | **8,000,000** |
| `batch_size` | 4096 (16 envs × 256 steps) |
| `learning_rate` | 1e-4 |
| `encoder_lr` | 1e-5 → 0 (cosine) |
| `encoder_snapback` | threshold=0.5, patience=100, min_reward=**0.05** |
| `sem_aux_coef` | 0.08 |
| `sem_pre_vq` | True |
| `sem_aux_start_reward` | 0.0 |
| `sem_class_weights` | enabled |

**Success criterion:**
- RL reward: any positive improvement (DoorKey-16x16 is much harder)
- Semantic probe overall > 70%
- Semantic probe goal > 50% (baseline = 0% without v6 encoder)
- All 6 classes represented in codebook (no zero-accuracy classes)

**Results:**

| Metric | Value |
|---|---|
| Best rolling avg reward (10-ep window) | — |
| Final 10-ep avg | — |
| Semantic probe overall | — |
| Semantic probe goal | — |
| Semantic probe door | — |
| Semantic probe key | — |
| Semantic probe wall | — |
| Semantic probe empty | — |

---

## Observations

### Core failure mode: encoder drift
Policy collapse is universal in all e2e experiments — the VQVAE encoder keeps receiving PPO gradients past the performance peak. The codebook EMA chases drifted encoder outputs, breaking the policy's input distribution. Every technique to slow encoder drift (lower LR, cosine decay, EMA target) merely delayed collapse; none stopped it.

### What actually fixed local performance (e2esnapback)
Two bugs in the vectorized rollout code caused all local experiments to peak at 0.29–0.66 despite the HPC run reaching 0.897:
1. **Trajectory length**: `batch_size=512 / num_envs=16 = 32 steps/env` — all rollouts truncated mid-episode. Fix: `batch_size=4096 → 256 steps/env`.
2. **Cross-env GAE contamination**: transitions stored in time-interleaved order made PPO's GAE backward pass use adjacent-env deltas as temporal successors. Fix: env-major ordering `[N, T, ...]` with `gamma=0` at env boundaries.

After both fixes: peak jumped from 0.473 → **0.9988**, exceeding e2estable's 0.897.

### What worked and what didn't
| Technique | Effect |
|---|---|
| Separate encoder LR (3e-5) | Helped — single biggest factor — allows policy to learn faster than encoder drifts |
| GAE (λ=0.95) + grad clipping + ortho init | Helped — PPO stability, required for high peaks |
| Cosine encoder LR decay | Helped — reduces final collapse but caps peak at 0.798 |
| Reconstruction loss (e2erecon) | Hurt — competed with RL, worst result (0.283 peak) |
| Hard freeze at 2.5M (e2ephased) | Failed — representation already degraded before freeze point |
| EMA target encoder | Failed — target follows online drift with delay — doesn't stop it |
| **Snapback** | Helped — prevented total collapse (0.000 → 0.3995 final) but Phase 3 too short (840k steps) |
| **GAE bugs fixed** | Helped — unlocked full performance — peak 0.9988, new best |
| **10M training budget** | Helped — encoder never collapsed; final 0.9984, overall 0.9732 — **new best overall** |
| **EMA target encoder (τ=0.995)** | Failed — smooths short-term drift but online encoder still drifts long-term; peak 0.3987, final 0.0000 |

### Open question
With a frozen encoder (e2esnapback Phase 3), the policy trained for only 840k steps and reached 0.3995. This suggests the stable-encoder phase needs more budget. `vqvae_pretrain_ppo` tests the extreme case: all 5M steps are pure policy training on a reconstruction-trained frozen VQVAE — but this failed (peak 0.1927) because reconstruction-trained representations are not task-relevant.

---

## Summary Comparison

| # | Experiment | Hardware | Status | Best Reward | Final Avg | Key Change |
|---|---|---|---|---|---|---|
| 1 | e2ebaseline | gpu04 (SLURM) | Done | 0.397 | 0.000 | Baseline |
| 2 | e2estable | gpu04 (SLURM) | Done | 0.897 | 0.000 | PPO stability + separate encoder LR |
| 3 | e2erecon | gpu05 (SLURM) | Done | 0.283 | 0.000 | + recon loss (hurt performance) |
| 4 | e2ecosine | gpu04 (SLURM) | Done | 0.698 | 0.198 | + cosine LR decay (buggy) |
| 5 | e2ephased | gpu04 (SLURM) | Done | 0.594 | 0.000 | + hard freeze at 2.5M |
| 6 | e2ecosine2 | gpu04 (SLURM) | Done | 0.798 | 0.188 | bug fix: cosine only on encoder |
| 7 | e2eema | RTX 4090 (local) | Done | 0.663 | 0.000 | EMA target encoder — two GAE bugs masked real performance |
| 8 | e2esnapback | RTX 4090 (local) | Done | **0.9988** | 0.3995 | Fixed 2 GAE bugs + snapback; snapback triggered at 83% leaving only 840k steps to recover |
| 9 | e2esnapback_10m | RTX 4090 (local) | Done | **0.9988** | **0.9984** | 10M steps; snapback never triggered; encoder stable throughout; **new best overall** |
| 10 | e2e_ema_tau | RTX 4090 (local) | Done | 0.3987 | 0.0000 | EMA target encoder (τ=0.995); smoothed rollout repr but online encoder still drifted; collapsed to 0 final |
| 11 | vqvae_pretrain_ppo | RTX 4090 (local) | Done | 0.1927 | 0.0000 | Pretrained VQVAE frozen from step 0; reconstruction-trained encoder failed — random-policy data lacks goal coverage |
| 12 | vqvae_preinit_snapback_ppo | RTX 4090 (local) | Done | **0.9988** | **0.8988** | Same pretrained VQVAE init + controlled e2e finetune (`encoder_lr=1e-5`, cosine, snapback) |
| 13 | vae_wm_rl_v1 | RTX 4090 (local) | Done | 0.0922 (train) | 0.000 (real-eval) | World model pipeline: VAE + continuous transition + PPO in latent world (`rl_train_steps=300k`); poor real-env transfer |
| 14 | vae_wm_rl_v2 | RTX 4090 (local) | Done | 0.0699 (train) | 0.000 (real-eval) | VAE world model retry with data/IO fixes + stronger transition + shorter WM RL horizon; still poor transfer |
| 15 | vqvae_wm_rl_v2_bfs | RTX 4090 (local) | Done | 5.56 (train) | 0.000 (real-eval) | Rebuild replay with BFS (denser success signal), then VQVAE + discrete transition + PPO in world model; transfer still failed |
| 16 | vqvae_wm_rl_v3_shortroll_inpolicy | RTX 4090 (local) | Done | 0.263 (train) | 0.000 (real-eval) | Mixed BFS + policy replay, shorter transition horizon, shorter PPO rollout; transition improved but transfer still zero |
| 17 | vae_wm_rl_v3_actionfix_seq | RTX 4090 (local) | Done | 0.0495 (real-eval) | 0.0495 (real-eval) | Fixed runner bug so VAE world-model baseline actually trained sequential pipeline |
| 18 | vae_wm_rl_v4_shortroll | RTX 4090 (local) | Done | 0.2486 (real-eval) | 0.2486 (real-eval) | Matched PPO imagined horizon to transition horizon (`rl_unroll_steps=8`); transfer improved but remained weak |
| 19 | vae_wm_rl_v5_curriculum | RTX 4090 (local) | Done | 0.0000 (real-eval) | 0.0000 (real-eval) | Stronger continuous transition (`512x5`, `n_train_unroll=12`) + staged PPO horizons `8 -> 12 -> 16`; world-model reward improved but real transfer collapsed |
| 20 | vae_wm_rl_v9_transfer_gap | CPU (local) | Done | 0.0397 (real-eval) | — | Diagnostic: confirmed compounding reward optimism in continuous world model as main transfer failure |
| 21 | vae_wm_rl_v11_conservative_reward_longshort | RTX 4090 (local) | Running | — | — | Conservative reward penalties + longer short-horizon PPO budget |
| 22 | vae_encoder_frozen_ppo_v1 | RTX 4090 (local) | Running | — | — | Frozen VAE encoder + real-env PPO to test encoder quality |
| 23 | encoder_localctx_vqvae_v1 | RTX 4090 (local) | Running | — | — | local_ctx_vqvae encoder training + validation |
| 24 | vqvae_wm_compact_v1 | RTX 4090 (local) | Done | 0.000 (real-eval) | 0.000 | filter_size=3 (9 tokens): transition flat (0.21→0.21 ✅) but encoder too coarse — agent moves don't cross token boundaries, PPO starved (ep_rew_mean 0.07–0.14) |
| 25 | vqvae_wm_compact_v2 | RTX 4090 (local) | Done | 0.000 (real-eval) | 0.000 | filter_size=5 ✅ good repr; rl_unroll_steps=20 ❌ too long — OL loss 62× worse at h=10; PPO navigated corrupted hallucinated states |
| 26 | vqvae_wm_compact_v3 | RTX 4090 (local) | Done | 0.000 (real-eval) | 0.000 | filter_size=5 + rl_unroll_steps=5; still 0 transfer — confirmed pure imagination cannot bridge to reality |
| 27 | vqvae_wm_dyna_v1 | RTX 4090 (local) | Done | 0.000 (real-eval) | 0.000 | **Dyna-style**: 8 real + 16 imagined envs; Phase 1 all 0.000; Phase 2 policy collapse (negative transfer from Dyna weights) |
| 28 | mf_frozen_vqvae_v2 | RTX 4090 (local) | Done | 0.000 | 0.000 | Frozen VQVAE + SB3 PPO, 24 real envs, 5M steps — replicated exp #11 (wrong approach), not exp #12 |
| 29 | mf_e2e_snapback_v1 | RTX 4090 (local) | Done | 0.9988 | 0.9984 | **Exact replication of exp #12**: pretrained VQVAE + e2e PPO, encoder_lr=1e-5, cosine, snapback, filter_size=9, 81 tokens |
| 30 | mf_e2e_wm_aux_v1 | RTX 4090 (local) | Done | 0.9988 | 0.9985 | **World model auxiliary loss** (coef=0.1): discrete trans model trained online, next-state prediction shapes encoder. Neutral vs baseline. |
| 31 | mf_e2e_wm_aux_v2 | RTX 4090 (local) | Done | 0.9988 | 0.9986 | **Remove redundant standalone trans model training**: aux loss in PPO only, no separate optimizer step. |
| 32 | mf_e2e_wm_doorkey_v1 | RTX 4090 (local) | Done | **0.9988** | **0.9979** | **New env**: MiniGrid-DoorKey-8x8-v0; random-init VQVAE; wm aux coef=0.1; **best overall avg across all experiments** — world model benefit confirmed on harder task |
| 33 | mf_e2e_semantic_probe | RTX 4090 (local) | Done | **0.9988** | **0.9985** | **Semantic aux loss** (coef=0.005, gate=0.3) + wm aux (coef=0.1): SUCCESS after 2 fixes (layout bug + coef 10× reduction + reward gate). Gate fired at 0.3945. Probe: 85.2% (wall 93%, lava 46%, goal 0%). |
| 34 | mf_e2e_semantic_aux_v2 | RTX 4090 (local) | Done | **0.9987** | **0.9984** | **Improved semantic aux**: class weights + coef 0.01 + gate 0.1. RL identical but probe did NOT improve (84.4% vs 86.0%). Lava/goal/agent still poorly separated — class weighting alone insufficient. |
| 35 | mf_e2e_semantic_doorkey | RTX 4090 (local) | Done | **0.9988** | **0.9943** | **DoorKey + SemanticHeadV2**: pos encoding + 3×3 local conv + focal loss (γ=2) + coef 0.05. RL matched baseline but probe WORSE (58.4% vs 62.4%). Bottleneck is encoder, not head. |
| 36 | mf_e2e_semantic_doorkey_v5enc | RTX 4090 (local) | Done | **0.9988** | **0.9984** | **BREAKTHROUGH — Strided encoder v5**: no AdaptiveAvgPool2d, 8×8=64 tokens (1:1 grid alignment). Probe 82.9% (was 58–63%). Key 100%, door 94%, agent 100%, wall 97%. Goal still 0%. |
| 37 | mf_e2e_semantic_doorkey_v5enc_prevq | RTX 4090 (local) | Done | **0.9988** | **0.9982** | **Pre-VQ semantic loss** + codebook 256. Probe 83.5% (marginal +1.4%). Goal still 0% — loss signal alone cannot force encoder to separate goal. |
| 38 | mf_e2e_semantic_doorkey_v6enc_goal | RTX 4090 (local) | Done ✅ | **0.9988** | **0.9528** | **BREAKTHROUGH — Goal 0%→98.8%!** Color+coord v6 encoder. Probe: 91.1% overall (wall 99%, key 100%, door 99%, goal 98.8%, agent 100%). RGB shortcut provides per-tile color that VQ can't wash out. |
| 39 | mf_e2e_semantic_doorkey_v7enc_multiscale | RTX 4090 (local) | Done ✅ | **0.9988** | **0.9984** | **v7 multi-scale+SE encoder**: goal retained (94%) but wall collapsed (96→78%). Multi-scale skips hurt trunk. Overall 78.3%. |
| 40 | mf_e2e_semantic_doorkey_v8enc_gated | RTX 4090 (local) | Done ✅ | **0.9988** | **0.9972** | **v8 gated input skip**: goal retained (95%) but empty/wall collapsed. Gating couldn't selectively help goal without hurting trunk. Overall 71.0%. |
| 41 | mf_e2e_semantic_doorkey_v9enc_patch | RTX 4090 (local) | Done ✅ | **0.9988** | **0.9981** | **v9 ViT-style patch embedding**: goal returned to 0%! Disproved receptive-field contamination hypothesis. Problem is VQ codebook allocation, not architecture. Overall 78.0%. |
| 42 | mf_e2e_semantic_doorkey16_v6enc_restart | RTX 4090 (local) | Done ✅ | **0.9988** | **0.2716** | **DoorKey-16x16 scaling test**: v6 encoder + VQ dead-code restart (codebook 512, 256 tokens). Goal partially preserved (51.7%) but wall collapsed (20%). Harder env needs more capacity. |
| 43 | mf_e2e_semantic_crafter_v6enc | RTX 4090 (local) | Pending | — | — | **Crafter (non-MiniGrid)**: v6 encoder on 64×64 Crafter obs, 19 semantic classes, codebook 512, 8M steps. Tests generalization of color-coord shortcut to 2D survival game. |

---

## Conclusion

### What was solved

The core goal — a VQVAE encoder trained end-to-end with PPO that achieves near-optimal policy performance — is solved. The best recipe (established by experiments 9, 12, 29):

1. **Pretrain VQVAE on reconstruction** from random-policy data to give the encoder a good starting point.
2. **Fine-tune e2e with conservative encoder LR** (`encoder_lr = 1e-5`, 10× lower than policy LR).
3. **Cosine anneal the encoder LR** to zero, preventing late-training drift without capping the peak.
4. **Snapback safety valve**: save the best encoder state; if rolling reward drops > 50% from peak, restore and hard-freeze. Prevents total collapse if drift eventually occurs.
5. **Correct vectorized rollout** (`batch_size = num_envs × ≥256` for sufficient per-env trajectory length, env-major GAE ordering to prevent cross-env contamination).

This recipe reliably reaches **0.9984–0.9988 final reward** on MiniGrid-LavaCrossing and MiniGrid-DoorKey-8x8 within 5–10M steps, with no collapse.

### What was not solved: world-model transfer

Experiments 13–27 represent a systematic attempt to train a latent world model and transfer the learned policy to the real environment. Every approach failed (0.000 real-env transfer) until Dyna-style real-env grounding was added — and even Dyna produced negative transfer. The root causes:

- **Reward hallucination**: continuous world models develop compounding reward optimism along imagined rollouts. PPO optimizes the imagined objective, not the real one.
- **Transition compounding error**: discrete VQVAE transition models are accurate teacher-forced at 1-step but diverge rapidly open-loop (97.9% accuracy at 3 steps → catastrophic at 10+).
- **Off-policy distribution mismatch**: transition models trained on replay data don't generalise to the states the learned policy visits.

No architectural improvement (BFS data, shorter horizons, curriculum, Dyna, stronger transitions) resolved these issues. The world-model line is a dead end without a fundamentally different approach (e.g., MBPO-style real-data interleaving, Dreamer-style RSSM, or online transition model updating).

### What was learned about encoder semantic grounding (experiments 33–42)

The spatial structure of the VQVAE encoder critically determines what information the codebook preserves:

| Encoder variant | Key property | Overall probe |
|---|---|---|
| v2 (`AdaptiveAvgPool2d(9×9)`) | Upsamples 5×5→9×9, blurs cell boundaries | 56–63% |
| v5 (strided conv, 8×8) | 1:1 cell alignment, no pooling | 82–83% |
| v6 (strided + pooled RGB + coords) | Explicit per-cell color shortcut | **88–91%** |
| v7 (multi-scale skips + SE) | Over-engineered; hurt trunk features | 78% |
| v8 (gated input skip) | Gate changes trunk gradient flow | 71% |
| v9 (ViT-style patch embed) | Independent per-tile; VQ still collapses rare codes | 78% |

**Key findings:**
- `AdaptiveAvgPool2d` with upsampling destroys spatial boundaries — replacing it with strided convolutions at the natural output resolution is a prerequisite for semantic grounding.
- Goal (1.6% of tokens) is the hardest class. It requires the encoder to produce geometrically distant pre-VQ features so the VQ codebook is forced to allocate a separate code. This was achieved only with an explicit per-tile pooled-RGB shortcut (v6).
- The VQ layer is a hard bottleneck for rare classes: even if the encoder produces distinct features, majority-class codes dominate EMA updates and rare classes get absorbed. VQ dead-code restart partially mitigates this but is not a full solution.
- Adding auxiliary losses (world model, semantic) is neutral-to-slight-positive on RL performance when coefficients are kept small (`≤0.1`). Large coefficients (> 0.05 on semantic) destabilise the encoder.

### Open directions

1. **Semantic grounding for all classes**: v6 solves goal but regresses on door/key vs v5. A combined architecture preserving v5's full-width trunk while adding the RGB shortcut selectively (not globally fused) may recover all classes simultaneously.
2. **DoorKey-16x16**: Experiment 42 showed the v6+restart approach partially transfers to a harder environment (goal 51.7%) but degrades wall badly (20%). Encoder capacity needs to scale with grid size.
3. **World model**: The discrete VQVAE + MLP transition architecture hit a ceiling. Sequence-model-based transitions (Transformer, RSSM) operating directly on codebook indices might improve multi-step fidelity. Alternatively, MBPO-style interleaving of real and imagined rollouts (rather than pure imagination) would eliminate the reward hallucination problem entirely.
4. **Crafter generalization** (exp 43, pending): Test whether the v6 color+coord encoder and semantic aux pipeline transfers to a non-grid-world visual environment with 19 semantic classes.
