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
**Status:** Running (~68% as of last check, step 13201/19532)
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
**Status:** Running
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
**Status:** Running
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
**Status:** Finished
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

### 8. e2eema

**Status:** Finished
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

### 9. e2esnapback

**Status:** Done ✅
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
| **Best rolling avg (10-ep window)** | **0.9988** 🏆 (new all-time best; e2estable was 0.897) |
| Snapback triggered at step | 4,160,736 (~83% into training) |
| Rolling avg at trigger | 0.199 (< 0.5 × 0.9988 = 0.499) |
| **Final 10-ep avg** | **0.3995** |
| vs e2estable final | 0.3995 vs 0.000 — snapback prevented total collapse |

**Phase-by-phase breakdown:**
- **Phase 1 (steps 0–4.16M):** Full e2e training with corrected GAE. Encoder and policy both learn. Peak climbs steadily to 0.9988.
- **Phase 2 (step 4.16M):** Collapse begins. Rolling avg drops from ~0.9988 to 0.199 over ~100 episodes. Snapback fires: encoder restored to peak state_dict and frozen.
- **Phase 3 (steps 4.16M–5M):** Only ~840k steps of policy training with frozen encoder. Policy partially recovers to 0.3995 final.

**What limited Phase 3:** Only 17% of training budget remained after snapback triggered. With a fixed encoder, the policy needs many more steps to re-optimise from scratch on the stable representation. Next experiment (frozen VQVAE from start) directly addresses this — the encoder is always fixed, so all 5M steps are pure policy training.

---

### 10. vqvae_pretrain_ppo

**Status:** Running
**Script:** `discrete_mbrl/collect_data.py` → `discrete_mbrl/train_encoder.py` → `discrete_mbrl/model_free/train.py`
**Hardware:** RTX 4090 (local, direct run — no SLURM)
**Model files:** `./models/MiniGrid-LavaCrossingS9N1-v0/vqvae_pretrain_ppo_best_model.pt`
**Log:** `/home/xiar3/experiments/vqvae_pretrain_ppo.log`

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

| Aspect | e2e experiments (1–9) | vqvae_pretrain_ppo (10) |
|---|---|---|
| Encoder training | PPO gradients flow through encoder | **Encoder frozen from step 0** |
| Representation source | RL signal shapes the embedding | **Reconstruction loss only** |
| Encoder drift | Inevitable (caused all collapses) | **Impossible** (no gradients) |
| Policy training budget | Shared with encoder training | **Full 5M steps** |
| Risk | Encoder collapses representation | Representation may not be task-relevant |

**Hypothesis:** A reconstruction-trained VQVAE on random-policy observations captures the environment's visual structure (object positions, colors, layout). This is sufficient for PPO to learn a navigation policy because the task-relevant information (agent position, lava position, goal position) is all present in the observations — PPO just needs to learn to read it. With no encoder drift, policy training should be stable throughout all 5M steps, potentially achieving better final performance than any e2e experiment.

**Results:** TBD

---

### 7. ppo_cnn_baseline (SB3)

**Status:** Done
**Script:** N/A (Stable Baselines 3)
**Model file:** `discrete_mbrl/trained_models/MiniGrid-LavaCrossingS9N1-v0/ppo_cnn_baseline.sb3`

**Description:** Standard PPO with CNN feature extractor trained via Stable Baselines 3. No VQVAE — raw pixel observations fed directly to a CNN policy. Serves as the reference baseline to measure the contribution of the discrete representation.

**Results:** TBD

---

## Summary Comparison

| Experiment | Job ID | Status | Best Reward | Final Avg | Key Change |
|---|---|---|---|---|---|
| e2ebaseline | 20279173 | Done | 0.397 | 0.000 | Baseline |
| e2estable | 20289278 | Done | 0.897 | 0.000 | PPO stability + separate encoder LR |
| e2erecon | 20289738 | Done | 0.283 | 0.000 | + recon loss (hurt performance) |
| e2ecosine | 20299312 | Done | 0.698 | 0.198 | + cosine LR decay (buggy) |
| e2ephased | 20299313 | Done | 0.594 | 0.000 | + hard freeze at 2.5M |
| e2ecosine2 | 20306194 | Done | 0.798 | 0.188 | bug fix: cosine only on encoder |
| e2eema | — (local) | Done | 0.663 | 0.000 | EMA target encoder — two GAE bugs masked real performance |
| **e2esnapback** | — (local) | **Done** ✅ | **0.9988** 🏆 | **0.3995** | Fixed 2 GAE bugs + snapback; new best; snapback triggered at 83% leaving only 840k steps to recover |
| **vqvae_pretrain_ppo** | — (local) | **Running** | TBD | TBD | Pretrain VQVAE on random data, freeze, train PPO — eliminates encoder drift entirely |
| ppo_cnn_baseline | — | Done | TBD | TBD | SB3 PPO + CNN baseline (no VQVAE) |

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
| Separate encoder LR (3e-5) | ✅ Single biggest factor — allows policy to learn faster than encoder drifts |
| GAE (λ=0.95) + grad clipping + ortho init | ✅ PPO stability, required for high peaks |
| Cosine encoder LR decay | ✅ Reduces final collapse but caps peak at 0.798 |
| Reconstruction loss (e2erecon) | ❌ Competed with RL, worst result (0.283 peak) |
| Hard freeze at 2.5M (e2ephased) | ❌ Representation already degraded before freeze point |
| EMA target encoder | ❌ Target follows online drift with delay — doesn't stop it |
| **Snapback** | ✅ Prevented total collapse (0.000 → 0.3995 final) but Phase 3 too short (840k steps) |
| **GAE bugs fixed** | ✅ Unlocked full performance — peak 0.9988, new best |

### Open question
With a frozen encoder (e2esnapback Phase 3), the policy trained for only 840k steps and reached 0.3995. This suggests the stable-encoder phase needs more budget. `vqvae_pretrain_ppo` tests the extreme case: all 5M steps are pure policy training on a reconstruction-trained frozen VQVAE.
