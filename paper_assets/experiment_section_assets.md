# Experiment Section Asset Plan

This file lists the tables and visualizations required to support the current experiment section draft. It is organized in the same order as the section so the assets can be inserted directly into the paper.

## Section 4.2 Environments

### Figure 1. Environment overview

**Type:** diagram or panel figure

**Purpose:** introduce the two evaluation environments before the quantitative results.

**Content:**
- One representative frame from `MiniGrid-LavaCrossingS9N1-v0`
- One representative frame from `MiniGrid-DoorKey-8x8-v0`
- Short labels indicating the task-critical objects in each environment
  - LavaCrossing: goal, lava, agent
  - DoorKey: key, door, goal, agent

**Why it is needed:** the experiment section refers to the environments as complementary settings. A visual makes that immediately legible to the reader.

## Section 4.3 Compared Models

### Table 1. Model variants and architectural differences

**Type:** summary table

**Purpose:** clarify what changes between `WM-only`, `Semantic aux`, `v5enc`, `v5_prevq`, `v6_goal`, `v7_multiscale`, and `v8_gated`.

**Columns:**
- Variant
- Environment
- Latent grid size
- Codebook size
- Semantic objective
- Goal-aware supervision
- Special architectural change

**Why it is needed:** the current experiment text names several variants, but a reader needs a compact reference table to keep them straight.

## Section 4.4 Evaluation Metrics / 4.5 Probe Protocol

### Table 2. Evaluation protocol summary

**Type:** compact setup table

**Purpose:** make the experimental protocol easy to scan.

**Columns:**
- Metric
- What is measured
- Reported statistics
- Applies to which experiments

**Rows:**
- Task reward
- Semantic probe accuracy
- World-model fidelity
- Transfer gap

**Why it is needed:** this reduces repeated prose and makes the setup more conference-friendly.

## Section 4.6 Main Results on Semantic Representation

### Table 3. LavaCrossing semantic probe results

**Type:** main quantitative table

**Status:** already generated

**File:** [lavacrossing_probe.md](/home/xiar3/experiments/STVqvae/paper_assets/tables/lavacrossing_probe.md)

**Purpose:** support the claim that generic semantic shaping improves common semantics but still fails on the goal.

**Columns:**
- Model
- Empty
- Wall
- Goal
- Lava
- Agent
- Overall

### Figure 2. LavaCrossing semantic probe accuracy

**Type:** grouped bar chart

**Status:** already generated

**File:** [lavacrossing_probe_accuracy.png](/home/xiar3/experiments/STVqvae/paper_assets/figures/lavacrossing_probe_accuracy.png)

**Purpose:** visually emphasize the classwise pattern, especially `0.0%` goal recovery despite high overall accuracy.

### Table 4. DoorKey semantic probe results

**Type:** main quantitative table

**Status:** already generated

**File:** [doorkey_probe.md](/home/xiar3/experiments/STVqvae/paper_assets/tables/doorkey_probe.md)

**Purpose:** support the main paper claim that the goal-aware variant changes the representation qualitatively.

**Columns:**
- Model
- Empty
- Wall
- Door
- Key
- Goal
- Agent
- Overall

### Figure 3. DoorKey semantic probe accuracy

**Type:** grouped bar chart

**Status:** already generated

**File:** [doorkey_probe_accuracy.png](/home/xiar3/experiments/STVqvae/paper_assets/figures/doorkey_probe_accuracy.png)

**Purpose:** visually show the jump from `0.0%` goal accuracy in earlier variants to high goal accuracy in `v6_goal` and related variants.

### Figure 4. Overall semantic probe accuracy summary

**Type:** side-by-side bar chart

**Status:** already generated

**File:** [overall_probe_accuracy.png](/home/xiar3/experiments/STVqvae/paper_assets/figures/overall_probe_accuracy.png)

**Purpose:** provide a high-level summary figure for readers who do not want to inspect every per-class result.

### Figure 5. Qualitative semantic reconstructions

**Type:** panel figure

**Purpose:** show representative GT vs predicted semantic grids for a few key models.

**Recommended panels:**
- LavaCrossing `WM-only`
- LavaCrossing `Semantic aux`
- DoorKey `WM-only`
- DoorKey `v5_prevq`
- DoorKey `v6_goal`

**What to highlight:**
- failure to represent sparse objects in baseline models
- better scene organization in semantic models
- correct goal localization in the goal-aware variant

**Why it is needed:** the logs already contain qualitative examples, and this figure will make the central representation argument much more concrete.

## Section 4.7 Results on Task Performance

### Table 5. Task reward summary across major variants

**Type:** reward summary table

**Purpose:** consolidate the main reward numbers used in the experiment section.

**Columns:**
- Model
- Environment
- Best rolling average reward
- Final 10-episode average
- Overall average reward
- Notes

**Important rows to include:**
- `mf_e2e_wm_doorkey_v1`
- `mf_e2e_semantic_doorkey`
- `mf_e2e_semantic_aux`
- `mf_e2e_semantic_aux_v2`
- older `experiments.md` entries that you cite in the text, if you keep them in scope

**Why it is needed:** the experiment section discusses reward, but currently there is no single asset that summarizes those numbers.

### Figure 6. Reward trajectory comparison

**Type:** line plot

**Purpose:** show that DoorKey is near ceiling even without semantics, and optionally show the instability/collapse pattern on LavaCrossing.

**Possible versions:**
- DoorKey only: `WM-only` vs semantic variants
- LavaCrossing only: baseline vs stable vs cosine/semantic variants
- Two-panel figure: DoorKey saturation and LavaCrossing instability

**Why it is needed:** a trajectory plot is more convincing than isolated final reward numbers.

## Section 4.8 World-Model Transfer Gap

### Table 6. LavaCrossing transfer-gap results

**Type:** diagnostic table

**Status:** already generated

**File:** [transfer_gap.md](/home/xiar3/experiments/STVqvae/paper_assets/tables/transfer_gap.md)

**Purpose:** support the claim that imagined returns are substantially more optimistic than real returns.

**Columns:**
- Setting
- Mean reward
- Mean length

### Figure 7. Transfer-gap plot

**Type:** line plot

**Status:** already generated

**File:** [transfer_gap_reward.png](/home/xiar3/experiments/STVqvae/paper_assets/figures/transfer_gap_reward.png)

**Purpose:** visually show reward inflation as rollout horizon increases.

### Table 7. Open-loop model fidelity by horizon

**Type:** diagnostic table

**Purpose:** complement the transfer-gap figure with model error statistics.

**Suggested columns:**
- Horizon
- Open-loop state MSE
- Reward absolute error
- Continuation bias
- Discounted return bias

**Source:** `wm_runs/vae_wm_rl_v9_transfer_gap/results.json`

**Why it is needed:** this lets you connect planning failure to quantitative model bias, not just final reward discrepancy.

## Minimal Asset Set

If you want the smallest complete set for the current experiment section, include:

1. Figure 1: Environment overview
2. Table 3: LavaCrossing semantic probe results
3. Table 4: DoorKey semantic probe results
4. Figure 3: DoorKey semantic probe accuracy
5. Table 5: Task reward summary
6. Table 6: Transfer-gap results
7. Figure 7: Transfer-gap plot

## Stronger Conference-Ready Set

If you want the experiment section to feel complete and polished, include:

1. Figure 1: Environment overview
2. Table 1: Model variant summary
3. Table 3: LavaCrossing probe table
4. Figure 2: LavaCrossing probe chart
5. Table 4: DoorKey probe table
6. Figure 3: DoorKey probe chart
7. Figure 5: Qualitative semantic reconstructions
8. Table 5: Task reward summary
9. Figure 6: Reward trajectories
10. Table 6: Transfer-gap table
11. Figure 7: Transfer-gap plot
12. Table 7: Open-loop fidelity diagnostics

## Current Gaps

Assets already available:
- LavaCrossing probe table and figure
- DoorKey probe table and figure
- Overall probe summary figure
- Transfer-gap table and figure

Assets still needed:
- Environment overview figure
- Model variant summary table
- Reward summary table
- Reward trajectory figure
- Qualitative semantic reconstruction figure
- Open-loop fidelity table
