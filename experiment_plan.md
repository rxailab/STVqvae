# Experiment Plan — Paper Submission

**GPU:** Single RTX 4090
**MiniGrid DoorKey-8x8 runtime:** ~3h per run (5M steps, ~6s/it, 814 batches)
**Crafter runtime:** ~10.5h per run (8M steps, ~19s/it, 1954 batches)

---

## Phase 1: CRITICAL — Multi-Seed DoorKey-8x8 (Exp 47–66)

**Goal:** 3-5 seeds for v2, v5, v6, v9 on DoorKey-8x8 → error bars for the encoder comparison table.

**Why these 4 encoders:**
- **v2** = baseline (AdaptiveAvgPool, 56-63% probe)
- **v5** = spatial alignment fix (82-83% probe, goal=0%)
- **v6** = RGB+coord shortcut (88-91% probe, goal=98.8%)
- **v9** = domain-agnostic ViT patches (78% probe) — shows generalizability

**Experiment matrix (4 encoders × 5 seeds = 20 runs = ~60h):**

| Exp | Encoder | Seed | Est. Time | Notes |
|-----|---------|------|-----------|-------|
| 47  | v2      | 1    | 3h | Baseline seed 1 |
| 48  | v2      | 2    | 3h | Baseline seed 2 |
| 49  | v2      | 3    | 3h | Baseline seed 3 |
| 50  | v2      | 4    | 3h | Baseline seed 4 (optional) |
| 51  | v2      | 5    | 3h | Baseline seed 5 (optional) |
| 52  | v5      | 1    | 3h | Spatial alignment seed 1 |
| 53  | v5      | 2    | 3h | Spatial alignment seed 2 |
| 54  | v5      | 3    | 3h | Spatial alignment seed 3 |
| 55  | v5      | 4    | 3h | (optional) |
| 56  | v5      | 5    | 3h | (optional) |
| 57  | v6      | 1    | 3h | RGB shortcut seed 1 |
| 58  | v6      | 2    | 3h | RGB shortcut seed 2 |
| 59  | v6      | 3    | 3h | RGB shortcut seed 3 |
| 60  | v6      | 4    | 3h | (optional) |
| 61  | v6      | 5    | 3h | (optional) |
| 62  | v9      | 1    | 3h | ViT patches seed 1 |
| 63  | v9      | 2    | 3h | ViT patches seed 2 |
| 64  | v9      | 3    | 3h | ViT patches seed 3 |
| 65  | v9      | 4    | 3h | (optional) |
| 66  | v9      | 5    | 3h | (optional) |

**Implementation needed:**
1. Add `--seed` arg to train.py (sets torch, numpy, env seeds)
2. Create a multi-seed sweep script that runs sequential or parallel jobs
3. After each run: auto-run semantic probe → save results to CSV

**Outputs:** Table with mean±std for each encoder × metric:
- RL reward (peak, final, overall avg)
- Semantic probe: overall accuracy, per-class accuracy (wall, floor, door, key, goal, agent)
- Codebook utilization (% active codes)

---

## Phase 2: CRITICAL — Codebook Usage Analysis (Exp N/A — analysis only)

**Goal:** For each encoder variant, produce a code-to-class assignment matrix showing:
- How many codes are allocated to each object class
- What fraction of codes are dead
- Entropy of the code-class assignment matrix (higher = more diverse)

**Method:** Load each best checkpoint, run 10K frames, collect (code_index, semantic_label) pairs, compute the joint distribution P(code, class).

**Implementation:**
1. Write `analyze_codebook_semantics.py` — loads checkpoint, collects data, produces:
   - Code-class confusion matrix heatmap
   - Per-class code count (how many unique codes map primarily to each class)
   - Dead code fraction
   - Codebook entropy: H(code | class) and H(class | code)
2. Run on existing v2, v5, v6, v9 DoorKey-8x8 best models (already saved)
3. Also run on Crafter best models

**Estimated time:** ~30 min total (analysis, no training)

---

## Phase 3: IMPORTANT — v6 Ablation (Exp 67–69)

**Goal:** Isolate which of v6's 3 components solved goal:
- (A) Pooled RGB shortcut
- (B) x/y coordinate injection
- (C) Narrower trunk (32ch vs 64ch)

| Exp | Config | What it tests |
|-----|--------|---------------|
| 67  | v5 + RGB shortcut only (64ch trunk, no coords) | Does RGB alone solve goal? |
| 68  | v5 + coords only (64ch trunk, no RGB) | Do coordinates alone solve goal? |
| 69  | v5 + both RGB+coords (64ch trunk — full-width) | Is the narrow trunk hurting door/key? |

**Implementation:** Create v6a/v6b/v6c encoder variants in model_construction.py.
**Time:** 3 runs × 3h = 9h

---

## Phase 4: IMPORTANT — v5 + Dead-Code Restart (Exp 70)

**Goal:** Test if dead-code restart (already implemented in Exp 44) fixes v5's goal=0% without needing v6's RGB shortcut.

**Hypothesis:** v5 has good alignment (82-83% probe) but goal gets absorbed into wall's code. Dead-code restart forces underused codes back into the active set — this might be sufficient to give goal its own code.

| Exp | Config | What it tests |
|-----|--------|---------------|
| 70  | v5 + dead_code_threshold=2.0 on DoorKey-8x8 | Does VQ fix alone rescue goal? |

**Time:** 1 run × 3h = 3h. **This is a clean causal test separating "encoder must produce distant embeddings" from "codebook EMA must be fixed".**

---

## Phase 5: IMPORTANT — World Model Accuracy vs Semantic Probe (analysis)

**Goal:** Show correlation between semantic probe accuracy and 1-step world model transition accuracy per class.

**Method:**
1. For each encoder variant (v2, v5, v6, v9), load the best checkpoint
2. Collect (state, action, next_state) tuples from fresh rollout
3. Predict next-state tokens with the world model
4. Measure per-class transition accuracy: for each semantic class c, what fraction of tokens at class-c positions are correctly predicted in the next step?
5. Plot: semantic probe accuracy (x) vs transition prediction accuracy (y) per class

**Implementation:** Write `analyze_wm_semantic_accuracy.py`
**Time:** ~1h total (analysis, no training)

---

## Phase 6: NICE TO HAVE — Scaling Analysis (Exp 71)

**Goal:** DoorKey-8x8 vs DoorKey-16x16 with v6 encoder, same hyperparams.

Already have: Exp 38 (8x8, probe=91.1%) and Exp 42 (16x16, probe=27.2%).
**Missing:** Error bars. Run Exp 42 config with 2 more seeds.

| Exp | Config |
|-----|--------|
| 71  | DoorKey-16x16, v6 encoder, seed 2 |
| 72  | DoorKey-16x16, v6 encoder, seed 3 |

**Time:** 2 runs × ~4h = 8h

---

## Phase 7: NICE TO HAVE — Continuous VAE Baseline (Exp 73)

**Goal:** Run same linear probe on a continuous VAE encoder (no VQ bottleneck) to quantify how much information VQ destroys.

**Method:**
1. Train a standard VAE (same encoder architecture as v5, but continuous latent) on DoorKey-8x8
2. Run semantic probe on continuous representations
3. Compare: VAE probe accuracy vs v5 vs v6

**Implementation:** The codebase already has `ae_model_type='ae'` support.
**Time:** 1 run × 3h + probe

---

## Phase 8: Crafter Consolidation

**Goal:** Clean Crafter results for paper (already have Exp 43-46).

**What to present:**
- Exp 43: baseline (7.8 reward, 42% probe = trivial grass)
- Exp 44: dead-code restart fix (7.17 reward, 100% codebook util, 6 classes >10% sem acc)
- Exp 45: sqrt class weights (8.2 reward, grass 72% + water 24% detected)
- Exp 46: power 0.75 weights + post-VQ (7.39 reward, sand 14% newly detected)

**Analysis needed:**
- Codebook usage analysis (same as Phase 2)
- Visualization: reconstructions showing what the VQ codes capture

**No additional training needed** — but 1-2 more seeds would strengthen the results (optional, 10.5h each).

---

## Timeline (sequential, single GPU)

| Day | Phase | Experiments | GPU Hours |
|-----|-------|-------------|-----------|
| 1   | Phase 2 + Phase 4 | Codebook analysis (0.5h) + v5+dead-code Exp 70 (3h) | 3.5h |
| 1   | Phase 1a | v2 seeds 1-3 (Exp 47-49) | 9h |
| 2   | Phase 1b | v5 seeds 1-3 (Exp 52-54) | 9h |
| 2   | Phase 3a | v6 ablation Exp 67 (RGB only) | 3h |
| 3   | Phase 1c | v6 seeds 1-3 (Exp 57-59) | 9h |
| 3   | Phase 3b | v6 ablation Exp 68 (coords only) | 3h |
| 4   | Phase 1d | v9 seeds 1-3 (Exp 62-64) | 9h |
| 4   | Phase 3c | v6 ablation Exp 69 (both, wide trunk) | 3h |
| 5   | Phase 5 | WM analysis (1h) + Phase 6 scaling (8h) | 9h |
| 5   | Phase 7 | VAE baseline Exp 73 (3h) | 3h |

**Total: ~5 days, ~72 GPU hours for all critical+important experiments.**

With optional 4th and 5th seeds: add ~24h (1 extra day).

---

## Implementation Priority

1. **Add `--seed` to train.py** — global reproducibility (torch, numpy, env)
2. **Write multi-seed sweep runner** — sequential batch script
3. **Write codebook analysis script** — Phase 2
4. **Create v6a/v6b/v6c encoder ablations** — Phase 3
5. **Write WM semantic accuracy analysis** — Phase 5
6. **Auto-probe script** — runs semantic probe after each training run completes
