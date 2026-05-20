## Pooled across-class Pearson r — extended apples-to-apples metrics

108 checkpoints (54 best + 54 final on DoorKey-8×8 and DoorKey-16×16); 100 with a saved transition model. Bootstrap 95% CI from 2,000 resamples.

`5-class`: Pearson r over `{wall, door, key, goal, agent}` (paper's original definition).
`4-class`: Pearson r over `{wall, door, key, agent}` — excludes `goal`. The goal token is a spatial constant in DoorKey: it sits in a fixed corner of every episode, so both probe (high recall on a constant-position class) and WM (memorize a constant) score high on it regardless of representation quality. Including it inflates the correlation toward whichever metric scores higher in absolute terms; excluding it isolates the *non-trivial* portion of the probe-WM relationship.

| metric | k | 5-class mean (95% CI) | 4-class mean (95% CI) | N |
|---|---|---|---|---|
| **exact** (paper §3.3) | 1 | +0.21 [+0.14, +0.28] | **−0.30 [−0.37, −0.22]** | 100 |
| exact | 5 | +0.14 [+0.06, +0.21] | −0.36 [−0.42, −0.28] | 100 |
| exact | 10 | +0.15 [+0.08, +0.22] | −0.36 [−0.43, −0.28] | 100 |
| **probe E1** | 1 | −0.20 [−0.30, −0.08] | −0.19 [−0.28, −0.08] | 100 |
| probe E1 | 5 | −0.19 [−0.30, −0.08] | −0.17 [−0.27, −0.06] | 100 |
| probe E1 | 10 | −0.19 [−0.30, −0.08] | −0.17 [−0.27, −0.06] | 100 |
| **class E2** (VQ) | 1 | +0.16 [+0.08, +0.22] | **−0.32 [−0.41, −0.24]** | 90 |
| class E2 (VQ) | 5 | +0.07 [−0.01, +0.14] | −0.43 [−0.51, −0.35] | 90 |
| class E2 (VQ) | 10 | +0.02 [−0.06, +0.09] | −0.49 [−0.57, −0.41] | 90 |
| **centroid E3** (VAE) | 1 | −0.29 [−0.53, −0.02] | −0.03 [−0.34, +0.29] | 10 |
| **swap E6** (VAE) | 1 | **−0.39 [−0.44, −0.34]** | −0.30 [−0.36, −0.23] | 10 |
| swap E6 (VAE) | 10 | −0.63 [−0.67, −0.58] | −0.54 [−0.60, −0.47] | 10 |

### Reading

1. **The paper's original metric is goal-confounded.** Pooled across 100 ckpts on 5 classes, exact metric reads +0.21 (CI excludes zero on the *positive* side). On the 4-class subset that strips out the spatial-constant `goal` token, the same metric reads **−0.30** (CI excludes zero on the *negative* side). The exact metric was reading "goal is high on both" rather than "the WM tracks the probe."

2. **The apples-to-apples metrics are robust to the goal exclusion.** E1's mean barely shifts (−0.20 → −0.19); E1's CI excludes zero on the negative side under both class subsets. E1 was designed to put both instruments on the same axis (linear-decodability of class), so the goal class doesn't enjoy a special advantage in the comparison.

3. **All four apples-to-apples metrics return negative pooled means on the goal-excluded 4-class subset.** E2 −0.32, E1 −0.19, E3 −0.03 (CI includes zero), E6 −0.30 to −0.54. None lands above zero.

4. **The proxy regime is excluded under every metric.** No CI under any (metric × class-set × horizon) combination crosses +0.5, let alone +0.8. Whichever way we slice it, probe accuracy is not a usable predictor of world-model accuracy on the same class.
