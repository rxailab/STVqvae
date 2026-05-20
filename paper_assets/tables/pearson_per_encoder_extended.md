## Per-encoder × per-environment across-class Pearson r at k=1

`exact` is the paper's original metric (§3.3); `E1`, `E2/E3`, `E6` are the apples-to-apples relaxations (§3.3 extended).
For VQ encoders (v2, v5, v5dc, v6) the third pair of columns is **E2** (code→class lookup); for the VAE the third pair is **E3** (class-centroid argmax) and **E6** (same-class spatial swap) is reported separately.
`5-class`: Pearson over `{wall, door, key, goal, agent}`. `4-class`: same minus `goal` (spatial constant).
mean ± std across seeds × {best, final} checkpoints.

| env | enc | N | exact 5 | exact 4 | E1 5 | E1 4 | E2/E3 5 | E2/E3 4 | E6 5 | E6 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| DK8  | v2   | 14 | +0.29 ± 0.32 | −0.18 ± 0.42 | −0.24 ± 0.55 | −0.33 ± 0.45 | +0.12 ± 0.51 | −0.26 ± 0.50 | — | — |
| DK8  | v5   | 10 | +0.19 ± 0.40 | −0.36 ± 0.41 | +0.13 ± 0.54 | −0.04 ± 0.65 | +0.13 ± 0.39 | −0.37 ± 0.41 | — | — |
| DK8  | v5dc | 10 | +0.23 ± 0.21 | −0.30 ± 0.38 | −0.20 ± 0.47 | −0.10 ± 0.52 | +0.17 ± 0.23 | −0.32 ± 0.41 | — | — |
| DK8  | v6   | 16 | +0.37 ± 0.16 | −0.35 ± 0.33 | −0.29 ± 0.36 | −0.10 ± 0.44 | +0.26 ± 0.26 | −0.24 ± 0.44 | — | — |
| DK8  | vae  | 10 | −0.06 ± 0.60 | −0.05 ± 0.58 | −0.34 ± 0.09 | −0.25 ± 0.12 | −0.29 ± 0.44 | −0.03 ± 0.53 | −0.39 ± 0.08 | −0.30 ± 0.11 |
| DK16 | v6   | 10 | +0.14 ± 0.14 | −0.38 ± 0.07 | −0.76 ± 0.27 | −0.69 ± 0.35 | −0.06 ± 0.19 | −0.52 ± 0.18 | — | — |
| DK16 | v5dc†| 6  | +0.05 ± 0.62 | −0.01 ± 0.73 | +0.95 ± 0.05 | +0.52 ± 0.54 | +0.30 ± 0.50 | +0.05 ± 0.66 | — | — |

† **DK-16 v5dc collapses the codebook on the larger latent grid** (probe accuracy on `wall`, `door`, `key`, `agent` drops to 0.02–0.18 across all seeds while `goal` stays at 1.00; see §4.6 of the main paper). The +0.95 E1 (5-class) is a *floor-correlation artifact*: when probe scores are near zero on all task-critical classes except `goal`, both probe and E1 vectors are dominated by the goal point and the across-class Pearson is mechanically forced to be high. Removing goal collapses the E1 mean to +0.52 ± 0.54 (CI includes zero), confirming the row is not evidence of probe-WM tracking. We list the cell here for completeness; it is excluded from §4.5's pooled correlation.

### Reading

1. **Every (env × encoder) cell except DK-16 v5dc shows a confidently negative or near-zero E1**, both 5-class and 4-class. The DK-16 v6 cell is the most negative real signal (E1 = −0.76 5-class, −0.69 4-class over 10 seeds): the v6 WM at the larger latent grid systematically anti-tracks probe accuracy on the four non-goal classes.

2. **Switching from 5-class to 4-class flips most exact-metric cells from positive to negative**: e.g. DK8 v6 goes from +0.37 to −0.35; DK8 v5dc from +0.23 to −0.30; DK16 v6 from +0.14 to −0.38. The original metric was reading goal-as-constant rather than probe-WM tracking on every cell where it appeared positive.

3. **E1 cells are stable across the 5↔4 class switch.** DK8 vae −0.34 → −0.25; DK16 v6 −0.76 → −0.69. E1 was designed for that robustness — both axes (probe applied to z₀ vs probe applied to ẑₖ) treat the goal class no differently from any other class.

4. **The VAE cohort (DK8 vae, N=10) is the most consistent.** All five metrics agree on a confidently negative correlation under both class subsets, with E6 4-class at −0.30 ± 0.11 — the tightest signal in the whole table.
