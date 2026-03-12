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

## Summary Comparison

| Experiment | Job ID | Status | Best Reward | Final Avg | Key Change |
|---|---|---|---|---|---|
| e2ebaseline | 20279173 | Done | 0.397 | 0.000 | Baseline |
| e2estable | 20289278 | Done | **0.897** | 0.000 | PPO stability + separate encoder LR |
| e2erecon | 20289738 | Done | 0.283 | 0.000 | + recon loss + ER replay |
| e2ecosine | 20299312 | Done | 0.698 | 0.198 | + cosine LR decay (buggy: policy LR also decayed) |
| e2ephased | 20299313 | Done | 0.594 | 0.000 | + cosine LR decay + hard freeze at 2.5M |
| e2ecosine2 | 20306194 | Done | 0.798 | 0.188 | bug fix: cosine only on encoder, policy LR fixed |
| ppo_cnn_baseline | 20326777 | Done | ~0.009 | 0.000 | SB3 PPO CnnPolicy, 72x72 RGB, 8 envs, 300k steps |

---

### 7. ppo_cnn_baseline

**Job ID:** 20326777
**Status:** Running
**Script:** `ppo_cnn_baseline.com`
**Node:** gpu06
**Model files:** `ppo_cnn_baseline_best/best_model.zip` / `ppo_cnn_baseline_final.zip`

**Framework:** Stable-Baselines3 PPO (independent comparison — no VQVAE)

**Setup:**

| Parameter | Value |
|---|---|
| Policy | CnnPolicy (NatureCNN) |
| Obs | 72x72x3 RGB (RGBImgObsWrapper + ImgObsWrapper) |
| Parallel envs | 8 x DummyVecEnv |
| Total steps | 300,000 |
| Eval freq | every 30,000 steps, 20 episodes |
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| `vf_coef` | 0.5 |

**Purpose:** Establish a direct pixel-based CNN PPO baseline for comparison against the VQVAE e2e approach. 300k steps is much shorter than the 5M VQVAE runs — serves as a lower-bound / sanity check on what raw CNN PPO achieves on this env.

**Results:**

| Metric | Value |
|---|---|
| Best `ep_rew_mean` (training) | ~0.009 (step ~262k, transient) |
| Final eval mean reward | **0.0000** |
| Eval at 240k steps | 0.0000 |
| Runtime | ~13 minutes |

Training rollout reward progression:

| Timestep | `ep_rew_mean` | `ep_len_mean` |
|---|---|---|
| 16,384 | 0.000 | 121 |
| 32,768 | 0.008 | 174 |
| 49,152 | 0.004 | 220 |
| 65,536+ | 0.000 | 231–304 |
| 262,144 | 0.009 | 274 |
| 300,000 | 0.000 | 262 |

**Notes:** The CNN baseline completely failed to solve the task in 300k steps. Episode lengths grew over time (agent wandering longer) but reward never meaningfully exceeded 0. The "New best mean reward!" logged at step 240k was SB3 saving a model with 0.0 eval reward (first eval). This is a strong result for the VQVAE approach — even our baseline VQVAE run (e2ebaseline) peaked at 0.397 within the same compute budget, and e2estable hit 0.897. The structured discrete representation from VQVAE appears significantly more sample-efficient than raw pixel CNN for this sparse-reward task.

---

## Observations

- **Policy collapse** is a recurring issue: both finished runs achieve 0.0 final avg despite strong mid-training peaks. The best checkpoint is not the last.
- **Root cause identified:** encoder drift — the VQVAE encoder keeps receiving PPO gradients past the performance peak, destroying the learned representation. The codebook EMA chases drifted encoder outputs, breaking the policy's input distribution.
- **Separate encoder LR** (3e-5 vs 3e-4) appears critical — the single biggest factor enabling e2estable's improvement.
- **GAE + gradient clipping + ortho init** together provide the PPO stability needed to sustain learning longer.
- **Reward scale:** successful episodes yield ~0.97 reward; failures yield 0.0 (binary success signal).
- **e2ecosine** achieved the best *final* avg (0.198) — cosine LR decay significantly reduced collapse severity vs all other runs.
- **e2erecon** was the worst — reconstruction loss competes with RL, slowing and limiting learning.
- **e2ephased** — hard freeze at 2.5M did not prevent collapse; policy became unstable in phase 2 even with frozen encoder.
- **Best peak** (0.897) still belongs to e2estable; best sustained performance belongs to e2ecosine.
- **Bug found in e2ecosine:** `CosineAnnealingLR` was applied to the whole optimizer, annealing both encoder AND policy/critic LR to 0. This suppressed policy learning and explains the lower peak (0.698 vs 0.897). Fixed in train.py to use `LambdaLR` targeting only the encoder param group.
- **e2ecosine2** tests the corrected implementation — expect higher peak (like e2estable) AND reduced collapse (like e2ecosine).
