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
**Status:** Running (sequential mode; active now)
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

| Aspect | e2e experiments (1–9) | vqvae_pretrain_ppo (10) |
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

### 11. vqvae_preinit_snapback_ppo

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

### 12. vae_wm_rl_v1

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

### 13. vqvae_wm_rl_v1

**Status:** Pending
**Pipeline:** `train_encoder.py` → `train_transition_model.py` → `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Logs:** `/home/xiar3/experiments/vqvae_wm_rl_v1_encoder.log`, `/home/xiar3/experiments/vqvae_wm_rl_v1_transition.log`, `/home/xiar3/experiments/vqvae_wm_rl_v1_rl.log`
**Model dir:** `./wm_runs/vqvae_wm_rl_v1`

**Setup:** VQVAE encoder (`codebook_size=64`, `embedding_dim=64`, `filter_size=9`) + discrete transition model + PPO in learned world model (`rl_train_steps=300000`).

**Results:** TBD

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

**Notes:** Reward density and world-model training signal improved significantly versus random-buffer runs, but real-environment transfer still failed (evaluation remained zero).

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
| **e2esnapback** | — (local) | **Done** | **0.9988** | **0.3995** | Fixed 2 GAE bugs + snapback; snapback triggered at 83% leaving only 840k steps to recover |
| **e2esnapback_10m** | — (local) | **Done** ✅ | **0.9988** 🏆 | **0.9984** 🏆 | 10M steps; snapback never triggered; encoder stable throughout; **new best overall** |
| **e2e_ema_tau** | — (local) | **Done** | **0.3987** | **0.0000** | EMA target encoder (τ=0.995); smoothed rollout repr but online encoder still drifted; collapsed to 0 final |
| **vqvae_pretrain_ppo** | — (local, RTX 4090) | **Done** | **0.1927** | **0.0000** | Pretrained VQVAE (`ea136dc...`) loaded + frozen from step 0; reconstruction-trained encoder failed — random-policy data lacks goal coverage; representation not task-relevant |
| **vqvae_preinit_snapback_ppo** | — (local, RTX 4090) | **Done** ✅ | **0.9988** | **0.8988** | Same pretrained VQVAE init + controlled e2e finetune (`encoder_lr=1e-5`, cosine, snapback) |
| **vae_wm_rl_v1** | — (local, RTX 4090) | **Done** | **0.0922** (train) | **0.000** (real-eval) | World model pipeline: VAE + continuous transition + PPO in latent world (`rl_train_steps=300k`); poor real-env transfer |
| **vqvae_wm_rl_v1** | — (local, RTX 4090) | **Pending** | TBD | TBD | World model pipeline: VQVAE + discrete transition + PPO in latent world (`rl_train_steps=300k`) |
| **vae_wm_rl_v2** | — (local, RTX 4090) | **Done** | **0.0699** (train) | **0.000** (real-eval) | VAE world model retry with data/IO fixes + stronger transition + shorter WM RL horizon; still poor transfer |
| **vae_wm_rl_v3_actionfix_seq** | — (local, RTX 4090) | **Done** | **0.0495** (real-eval mean) | **0.0495** (real-eval) | Fixed runner bug so the VAE world-model baseline actually trained sequential `encoder -> transition -> RL` instead of the wrong e2e path |
| **vae_wm_rl_v4_shortroll** | — (local, RTX 4090) | **Done** | **0.2486** (real-eval mean) | **0.2486** (real-eval) | Matched PPO imagined horizon to the transition horizon (`rl_unroll_steps=8`); transfer improved but remained weak |
| **vae_wm_rl_v5_curriculum** | — (local, RTX 4090) | **Done** | **0.0000** (real-eval mean) | **0.0000** (real-eval) | Stronger continuous transition (`512x5`, `n_train_unroll=12`) + staged PPO horizons `8 -> 12 -> 16`; world-model reward improved but real transfer collapsed |
| **vqvae_wm_rl_v2_bfs** | — (local, RTX 4090) | **Done** | **5.56** (train) | **0.000** (real-eval) | Rebuild replay with BFS (denser success signal), then VQVAE + discrete transition + PPO in world model; transfer still failed |
| ppo_cnn_baseline | — | Done | TBD | TBD | SB3 PPO + CNN baseline (no VQVAE) |

---

## Next Experiment Recommendation

### 16. vqvae_wm_rl_v3_shortroll_inpolicy

**Status:** Done
**Pipeline:** `collect_data.py (mixed BFS + policy rollouts)` → `train_transition_model.py` → `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vqvae_wm_rl_v3_shortroll_inpolicy`

**Why this was the next experiment:**
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

**Why this beats the other options right now:**
- Better PPO tuning alone is low value because the current evidence says PPO already learns *inside the flawed model*.
- More reward density alone already failed (`vqvae_wm_rl_v2_bfs`).
- A new encoder is lower priority because the current VQVAE already supports strong real-environment control.
- This experiment directly tests the most likely bottleneck: **model exploitation caused by compounding transition error and distribution mismatch**.

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

**Recommended next experiment:**
1. Add an **oracle-action / real-env latent consistency evaluation** for the trained transition model.
2. Measure latent prediction error along real trajectories induced by the strong real policy, not just replay-buffer averages.
3. Compare open-loop vs closed-loop rollout error over horizons 1/3/5/10.
4. If latent drift under policy rollouts is still large, move to a stronger transition architecture or planning-based control instead of more PPO-in-model runs.

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

**Recommended next concrete experiment:** `vqvae_wm_rl_v4_stronger_trans`

**Goal:** improve the transition model itself rather than continuing to tune downstream control.

**Proposal:**
1. Keep the strong pretrained VQVAE encoder and the mixed replay pipeline.
2. Replace the current discrete MLP transition model with a **stronger autoregressive / transformer-style discrete transition model** already supported in the codebase (`transformer` or `transformerdec` transition type).
3. Train and evaluate it first with the same real-policy consistency diagnostic:
   - teacher-forced and open-loop latent error on horizons `1/3/5/10`
4. Only if those numbers improve materially, rerun:
   - latent MPC (horizon 3 or 5)
   - optionally world-model PPO afterward

**Decision rule for v4:**
- If the stronger transition model reduces teacher-forced loss substantially and keeps open-loop error controlled through horizon 3–5, continue with MPC / control experiments.
- If not, the next change should be the **representation/state target** itself, not the controller.

**Results:**

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

However, even after converging for 20 epochs, `transformerdec` was still dramatically worse than the current discrete MLP transition model (`0.89 / 1.26 / 1.79 / 1.98` on the same horizons). So `v4` failed the first gate before downstream control.

**Diagnostic follow-up (completed):**

Ran the same real-policy latent consistency diagnostic on the `transformerdec` checkpoint.

| Horizon | Segments | Teacher-forced state loss | Teacher-forced state acc | Open-loop state loss | Open-loop state acc |
|---|---:|---:|---:|---:|---:|
| 1 | 848 | 1036.57 | 0.0045 | 1035.53 | 0.0045 |
| 3 | 798 | 1031.42 | 0.0050 | 1731.43 | 0.0028 |
| 5 | 748 | 1026.16 | 0.0052 | 1838.35 | 0.0026 |
| 10 | 623 | 1017.31 | 0.0053 | 1835.79 | 0.0031 |

**Conclusion from v4:**  
`transformerdec` is not a stronger replacement here. It is substantially worse than the discrete baseline both on replay-buffer test loss and on real-policy latent consistency. This means the next experiment should **not** be PPO or MPC on top of this model.

**Recommended next experiment:**  
Keep the current best discrete transition model as the control baseline and change the **state target / supervision**, not just the transition architecture. The most direct next test is:
1. predict a lower-entropy target than raw VQVAE token grid, or
2. add auxiliary supervision tied to controllable structure (agent position / orientation / goal progress if recoverable),
3. then rerun the same diagnostic before any controller work.

### 17. vae_wm_rl_v4_shortroll

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

### 18. vae_wm_rl_v5_curriculum

**Status:** Done
**Pipeline:** `train_encoder.py` -> `train_transition_model.py` -> `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vae_wm_rl_v5_curriculum`

**Why this is the next VAE experiment:**
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

### 19. vae_wm_rl_v8_curriculum_bestselect

**Status:** Planned
**Pipeline:** `train_encoder.py` -> `train_transition_model.py` -> `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vae_wm_rl_v8_curriculum_bestselect`

**Why this is the direct fix to v5:**
- `vae_wm_rl_v5_curriculum` only reported the final PPO policy, even though real-env transfer can peak and then regress during world-model training.
- RL checkpoints were also being written to the shared path `discrete_mbrl/models/MiniGrid-LavaCrossingS9N1-v0/ppo_world_model.zip`, which allows later experiments to overwrite earlier ones.

**Changes in this run:**
- Keep the same VAE + stronger continuous transition + staged PPO curriculum as `vae_wm_rl_v5_curriculum`.
- Add periodic real-environment evaluation during PPO training (`--rl_eval_freq`).
- Save per-experiment PPO checkpoints inside the experiment run directory.
- Select the best real-eval PPO checkpoint, not just the final one, as the canonical saved policy for the run.

**Primary question:**
- Did `v5` actually fail throughout training, or was the best transferable policy discarded by evaluating only the terminal PPO weights?

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

### 21. vae_wm_rl_v10_conservative_reward

**Status:** Planned
**Pipeline:** `train_encoder.py` -> `train_transition_model.py` -> `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vae_wm_rl_v10_conservative_reward`

**Why this is the next experiment:**
- `vae_wm_rl_v9_transfer_gap` showed that the main failure in `v5` is optimistic reward drift, not just latent state error.
- On real-policy trajectories, the transition model predicts increasingly positive imagined return despite zero actual return. PPO then exploits that fictitious reward channel.

**Changes in this run:**
- Keep the stronger VAE + continuous transition + staged PPO curriculum from `v5` / `v8`.
- Keep periodic real-environment PPO evaluation and best-checkpoint selection from `v8`.
- Add conservative reward penalties during continuous transition training:
  - `--trans_reward_overestimate_coef 2.0`
  - `--trans_reward_zero_target_coef 4.0`
  - `--trans_reward_zero_margin 0.0`
- Concretely:
  - penalize reward overestimation on all transitions, and
  - penalize any positive reward prediction on zero-reward targets even more strongly.

**Hypothesis:**
- If the transfer failure is driven by imagined reward hallucination, then directly suppressing false-positive reward predictions should reduce the world-model/real-environment value gap and improve PPO transfer.
- The key success criterion is not just lower transition loss, but a smaller transfer gap under the `v9` diagnostic and a higher best real-env PPO score.

### 22. vae_wm_rl_v11_conservative_reward_longshort

**Status:** Running
**Pipeline:** `train_encoder.py` -> `train_transition_model.py` -> `train_rl_model.py`
**Hardware:** RTX 4090 (local)
**Model dir:** `./wm_runs/vae_wm_rl_v11_conservative_reward_longshort`

**Why this is the next experiment:**
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

### 23. vae_encoder_frozen_ppo_v1

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

### 24. encoder_localctx_vqvae_v1

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
| **10M training budget** | ✅ Encoder never collapsed; final 0.9984, overall 0.9732 — **new best overall** |
| **EMA target encoder (τ=0.995)** | ❌ Smooths short-term drift but online encoder still drifts long-term; peak 0.3987, final 0.0000 |

### Open question
With a frozen encoder (e2esnapback Phase 3), the policy trained for only 840k steps and reached 0.3995. This suggests the stable-encoder phase needs more budget. `vqvae_pretrain_ppo` tests the extreme case: all 5M steps are pure policy training on a reconstruction-trained frozen VQVAE — but this failed (peak 0.1927) because reconstruction-trained representations are not task-relevant.

---

### 25. e2esnapback_10m

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

**Expected outcome:** Peak reward similar to e2esnapback (~0.9988), but final reward substantially higher than 0.3995 due to longer post-snapback consolidation.

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

### 26. e2e_ema_tau

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
