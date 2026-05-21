# STVqvae Handover — Next-Paper Status (as of 2026-05-21)

This document is the single source of truth for picking the next-paper work
back up. The original NeurIPS 2026 paper ("Probe–WM dissociation") is
closed; this file covers the *follow-up* paper attempt that started
2026-05-11 and reached a major fork in the road today (2026-05-21).

## TL;DR

The follow-up paper aimed to be **constructive** — diagnose why the §4
probe-WM dissociation happens, then propose a recipe that fixes it during
online PPO+WM training.

- **Diagnosis succeeded.** Phase VIII-A (a replay-style diagnostic) cleanly
  localised the failure mode to the **joint encoder/WM online training**,
  not the encoder representations themselves.
- **All 14 recipe interventions failed.** 8 phases of experiments —
  ε-greedy at multiple levels, episode-level injection, warmup curricula,
  encoder LR variants, snapback off, stop-gradient decoupling, two-buffer
  architecture — none produce robust per-class WM accuracy at training end.
- **The only reliable per-class WM accuracy** comes from the §4.8 OFFLINE
  protocol (Cell A: frozen encoder, fresh WM trained on random data for
  100 epochs).

**Decision pending:** pivot to a negative-result paper that formalises this
"joint training is intractable" finding, or attempt one more architectural
recipe (the candidate would be encoder pre-training on random buffer
followed by PPO finetune — explicitly Cell-A-as-init).

## Current job state (as of 2026-05-21 19:05 BST)

```
RUNNING
  21526603  v6_twobuf_s5   ~5 min in, batch ~13/1221
            Phase VIII-C third seed; expected to fail per established pattern.
            Decision needed: cancel or let it finish (8 GPU-hours).
QUEUED:    none
PENDING:   none
```

To check live state: `squeue -u $USER` (need `export PATH=/usr/shared_apps/packages/git-2.49.0/bin:$PATH` for git too).

## Empirical findings — comprehensive table

All experiments use **MiniGrid-DoorKey-8x8-v0** with the v6 VQVAE encoder
(64-token grid, 64-dim embeddings, codebook 64, dead-code threshold 2.0)
unless otherwise noted. PPO config matches §4 paper baseline.

| Cell | What | door WM₁ | Notes |
|---|---|---|---|
| §4 online (no injection) | Standard joint PPO+WM | 0.02 | The dissociation |
| Cell A oracle (offline random) | Frozen encoder + fresh WM on random buffer, 100 epochs | **0.94** | The ceiling. Replicable. |
| Cell C oracle (offline policy) | Same as A but policy buffer | 0.74 | Lower than A |
| Cell D F1a/F1b oracle | Mixed-step / branch action-uniform offline | 0.00 | Action diversity ≠ enough |
| Phase III ε=0.15 (online, fixed patch) | ε-greedy at vec_env | 0.65 | Seed 1; bimodal at seed level |
| Phase III randep p=0.5 (online) | Episode-level random | 0.05 | Snapback fires |
| Phase IV ε=0.50 (online) | ε-greedy s1 | **0.61** | First headline-looking result |
| Phase V s2 ε=0.50 | replication | **0.67** | Held up |
| Phase V s3 ε=0.50 | replication | 0.00 | Failed |
| Phase V ε=0.30 s1 | Pareto fill-in | 0.00 | Failed |
| Phase V ε=0.70 s1 | Pareto fill-in | 0.67 | Held |
| Phase V ε=0.15 s2 | Pareto fill-in | 0.00 | Variance high |
| Phase VI s4–s6 ε=0.50 | More replication | 0.00 / 0.00 / 0.05 | **All failed** |
| Phase VI s3 nosnap | Snapback diagnostic | 0.14 | Marginal — snapback not the cause |
| Phase VII warmup 1M | Delayed injection | 0.07 / 0.07 | Broke working seeds |
| Phase VII lower encoder LR (5e-6) | Less plasticity | 0.00 | Broke working seed |
| Phase VII no cosine LR | Flat encoder LR | 0.02 | Broke working seed |
| Phase VII ε=0.70 s2 | Higher injection | 0.00 | Bimodal even at high ε |
| **Phase VIII-A** (offline-on-saved-encoders) | s2_succ and s5_fail offline | **0.99 / 0.95** | **Both encoders fine** |
| Phase VIII-B stop-grad PPO→enc | Encoder only via WM aux | 0.00 / 0.02 / 0.00 | Broke working seed |
| **Phase VIII-C two-buffer** | Static random buffer + WM-only updates | **0.001 / 0.000** | Broke working seed; didn't rescue failure |

**Bottom line: success rate of online ε=0.50 across 6 seeds is 2/6
(33%). No intervention raises that rate. Stop-grad and two-buffer
*broke* the previously-working s1.**

## Key code changes (in this paper-attempt cycle)

Files modified — all committed in `f96f728` (2026-05-20).

- `discrete_mbrl/training_helpers.py` — new flags
  - `--explore_random_prob` (step-level ε-greedy)
  - `--random_episode_prob` (per-env episode-level random)
  - `--explore_warmup_steps` (delay injection)
  - `--ppo_stop_grad_encoder` (VIII-B)
  - `--random_wm_buffer_eps`, `--random_wm_per_update`,
    `--random_wm_batch_size` (VIII-C two-buffer)

- `discrete_mbrl/model_free/train.py`
  - Vec-env rollout: applies `explore_random_prob` per-step + per-env
    `random_episode_flags` for `random_episode_prob`
  - Rolling-reward window: excludes random-action episodes from
    `recent_rewards` so best-model save reflects actual policy quality
  - Phase VIII-C buffer pre-collection block (before main loop)
  - VIII-C extra `train_wm_only` calls after each `ppo.train(batch_data)`

- `discrete_mbrl/model_free/ppo.py`
  - `ppo_stop_grad_encoder=False` constructor arg
  - `states_for_ppo = minibatch['states'].detach()` when flag set
  - New method `train_wm_only(obs, acts, next_obs)` for the two-buffer arch

- `discrete_mbrl/analyze_wm_action_cond_v2.py` — per-class action-dep + ε-collapse + residual-rank metrics
- `discrete_mbrl/analyze_buffer_coverage.py` — class-change rates in random vs policy buffers
- `discrete_mbrl/train_oracle_wm_actuniform.py` — env-clone branch / mixed-step oracle (Cell D)

## Key script locations

```
scripts/
  phaseIII_*.sbatch  phaseIII_*.sh         (Phase III ε-greedy + episode-level)
  phaseIV_*.sbatch   run_phaseIV_v6.sh     (Phase IV Pareto: ε=0.25, ε=0.50)
  phaseV_*.sbatch    run_phaseV_v6.sh      (Phase V seed replication)
  phaseVI_*.sbatch   run_phaseVI_v6.sh     (Phase VI snapback-off, more seeds)
  phaseVII_*.sbatch  run_phaseVII_v6.sh    (Phase VII stabilizers)
  phaseVIII_*.sbatch run_phaseVIII_*.sh    (Phase VIII A/B/C)
```

Each `run_phase*.sh` takes positional args (run_name, seed, ...) so new
sweeps can call the runner directly without writing a new sbatch.

## Where the data is

```
discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/
  v6_eps050_s{1..6}_best_model.pt           (Phase IV + V + VI ε=0.50 baseline)
  v6_eps050_s3_nosnap_best_model.pt          (Phase VI snapback-off)
  v6_eps050_warmup1M_s{2,3}_best_model.pt   (Phase VII warmup)
  v6_eps050_lowlr_s2_best_model.pt           (Phase VII lower encoder LR)
  v6_eps050_nocos_s2_best_model.pt           (Phase VII no cosine)
  v6_eps050_sgenc_s{1,3,5}_best_model.pt     (Phase VIII-B stop-grad)
  v6_twobuf_s{1,3,5}_best_model.pt           (Phase VIII-C two-buffer)
  phaseVIII_A/oracle_randonly_s2_succeeded.pt (the offline diagnostic ckpts)
  phaseVIII_A/oracle_randonly_s5_failed.pt

logs/
  phaseIII/, phaseIV/, ..., phaseVIII_C/
    v6_*.log                                (training stdout)
    analyze_wm_semantic_accuracy_*_best.json (per-class WM_1 — paper-deciding)
    analyze_wm_action_cond_v2_*_best.json    (action-dep diagnostics)
    *_final.json                              (same analyses on final_model)
```

## Memory files index

```
~/.claude/projects/-mmfs1-storage-users-xiar3-exp-STVqvae/memory/
  MEMORY.md                       — index of all memory entries
  project_overview.md             — high-level project description
  project_thesis.md               — NeurIPS 2026 (original) paper thesis
  project_layout.md               — directory map
  project_experiment_logs.md      — old experiment log conventions
  project_next_paper.md           — original next-paper plan (2026-05-11)
  next_paper_outline.md           — section-by-section paper plan
  phaseI_findings.md              — two failure modes diagnostic (Phase I)
  phaseII_findings.md             — Cell C refuted data-starvation
  phaseIII_findings.md            — ε=0.15 first attempt (with patch bug)
  phaseIV_findings.md             — ε=0.50 step-level "works"
  phaseV_findings.md              — recipe is bimodal
  phaseVIII_A_findings.md         — replay diagnostic localizes failure
```

## Open decision points

1. **s5 in flight (5h GPU)**: cancel or let finish?
   - Recommendation: **cancel** if you trust the negative-result pivot;
     the third Phase VIII-C seed will almost certainly match s1/s3.
2. **Paper direction**:
   - (A) Pivot to negative-result paper. Substantive contribution, but
     less marketable than a "fix found" story.
   - (B) One more architectural attempt: Cell-A-style ENCODER PRETRAINING
     followed by PPO finetune with the encoder mostly frozen
     (encoder_lr → 1e-7 or full freeze). This is the "least joint" recipe
     remaining and hasn't been tested.
   - (C) Try Crafter or DK-16 to see if the bimodality is env-specific.

## How to resume next session (script)

```bash
# Activate the env
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export PYTHONPATH=/mmfs1/storage/users/xiar3/exp/STVqvae:$PYTHONPATH

# Git: add to PATH first
export PATH=/usr/shared_apps/packages/git-2.49.0/bin:$PATH
cd /mmfs1/storage/users/xiar3/exp/STVqvae
git log --oneline -5

# Check queue and latest results
squeue -u $USER
ls logs/phaseVIII_C/*.json

# Re-run any one of the experiments with a new seed
bash scripts/run_phaseVIII_C_v6.sh v6_twobuf_s7 7  # for example
```

## Repository

```
GitHub: https://github.com/rxailab/STVqvae
Branch: master
Last commit: f96f728 "Next-paper experiments: state-coverage interventions (Phase I-VIII)"
```

PAT for pushing: was rotated 2026-05-20. If the cached token in
`git remote -v` is expired, use a fresh PAT via:
`git remote set-url origin https://x-access-token:NEW@github.com/rxailab/STVqvae.git`

## One-paragraph "what to tell the next person"

> The follow-up paper to NeurIPS 2026 attempted to find a recipe that
> closes the §4 probe-WM dissociation during online PPO+WM training. Across
> 8 phases (~50 experiments, ~250 GPU-hours), 14 distinct interventions
> all fail to produce robust per-class WM₁ accuracy. The diagnostic phase
> (VIII-A) cleanly shows the encoders themselves are fine — frozen
> encoders fed to a fresh offline WM on random data achieve door WM₁ ≈
> 0.95 reliably. The bimodality is therefore in the joint online
> encoder/WM optimization, and surface interventions cannot reach it. The
> paper should pivot to a negative-result framing or commit to one more
> architectural attempt (encoder pretraining + frozen PPO finetune) before
> closing out.
