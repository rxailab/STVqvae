## Phase D: imagined-return rank correlation with actual return

For each ckpt with a working trained policy, we run 100 trained-policy episodes in the env, then for each starting state run the WM forward K steps using the policy's chosen actions and accumulate

$$\hat G_0 = \sum_{k=0}^{K-1} \left(\prod_{j=0}^{k-1} \hat\gamma_j\right)\hat r_k + \left(\prod_{j=0}^{K-1} \hat\gamma_j\right) V_\eta(\hat z_K)$$

using the WM's own per-step reward and discount heads ($\hat r_k$, $\hat\gamma_k$) plus a bootstrap with the trained value head $V_\eta$ on the final imagined state. We then compare $\hat G_0$ to the actual discounted return $G_0 = \sum_{t} \gamma^t r_t$ across episodes. The Pearson rank-correlation between the two is the planning-relevant quantity: a planner using the WM to rank candidate trajectories needs $\hat G_0$ and $G_0$ to be order-correlated, even if their absolute scales differ.

| encoder | N | succ | mean V(z\*₀) | mean G_real | rank_corr (K=10) |
|---|---|---|---|---|---|
| v6 | 5 | 65% | +0.67 | +0.55 | **+0.06 ± 0.18** |
| vae | 5 | 57% | +0.55 | +0.48 | **+0.07 ± 0.07** |
| v5dc | 2 | 91% | +0.84 | +0.77 | **−0.09 ± 0.26** |
| v2 | 5 | 91% | +0.84 | +0.78 | **+0.00 ± 0.21** |
| v5 | 3 | 91% | +0.84 | +0.78 | **−0.02 ± 0.10** |
| v9 | 3 | 89% | +0.84 | +0.76 | **−0.02 ± 0.15** |

**Pooled across 20 working-policy ckpts: rank_corr = +0.007 ± 0.154 at K=10. 17 of 20 ckpts (85%) have |rank_corr| ≤ 0.2.**

### Reading

- **Imagined-return ranking is uncorrelated with actual-return ranking** under every encoder family we tested. Pooled mean is essentially zero; per-encoder means are all in [−0.09, +0.07]; per-ckpt magnitudes never exceed 0.35.
- **The bias direction varies wildly** by encoder family — vae and v5dc severely under-shoot (mean bias −0.5 to −1.7), v5 and v9 catastrophically under-shoot (−7 to −8), v2 is mean-calibrated with huge variance — but the absolute bias is recoverable by calibration; the **rank-noise is not.**
- **A planner querying the WM to compare candidate actions receives noise.** This is the planning-impossibility result: regardless of how the imagined return is calibrated, it cannot tell the planner which trajectories are better than which.
- The trained value head V applied directly to the *true* initial state z\*₀ is well-calibrated: mean V(z\*₀) ≈ G_real to within 0.05 across all six encoder cells. The dysfunction is specific to the WM-rolled imagined return, not to the value head itself.

### Bottom line

The WMs jointly trained with PPO in standard MBRL are **planning-incoherent**: the imagined return computed by free-running the WM and accumulating its own reward and discount heads has rank correlation with actual environment return that is statistically indistinguishable from zero. A planner cannot use these WMs to rank candidate actions — the principal use case the WM was trained for.
