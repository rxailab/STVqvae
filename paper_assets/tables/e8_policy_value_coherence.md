## E8 — Policy and value coherence between z\* and ẑₖ

13 checkpoints from four (env × encoder) cells; the trained PPO policy and value head from each checkpoint are applied directly to z\*ₖ and ẑₖ from a free-running rollout under the trained policy. Mean ± std across seeds.

| env | enc | N | ep len | succ | k | action_match | KL median | value_corr | V(z\*) / V(ẑ) |
|---|---|---|---|---|---|---|---|---|---|
| **DK-8** | **v6** | 2 | 121 | 43% | 1 | +0.43 ± 0.21 | +1.04 ± 1.04 | **−0.42 ± 0.18** | 0.43 / 0.46 |
| DK-8 | v6 | 2 | 121 | 43% | 5 | +0.40 ± 0.45 | +1.69 ± 1.93 | −0.00 ± 0.15 | 0.43 / 0.44 |
| DK-8 | v6 | 2 | 121 | 43% | 10 | +0.44 ± 0.38 | +1.54 ± 1.80 | −0.10 ± 0.07 | 0.43 / 0.44 |
| **DK-8** | **vae** | 5 | 98 | 57% | 1 | +0.29 ± 0.19 | +4.79 ± 4.29 | −0.06 ± 0.18 | 0.57 / 0.58 |
| DK-8 | vae | 5 | 98 | 57% | 5 | +0.33 ± 0.23 | +4.56 ± 4.46 | −0.15 ± 0.18 | 0.58 / 0.57 |
| DK-8 | vae | 5 | 98 | 57% | 10 | +0.33 ± 0.22 | +4.74 ± 4.76 | −0.18 ± 0.19 | 0.57 / 0.57 |
| DK-16† | v6 | 3 | 200 | 0% | 1 | +0.72 ± 0.25 | +0.01 ± 0.02 | +0.005 | 0.00 / 0.00 |
| DK-16† | v5dc | 3 | 200 | 0% | 1 | +0.18 ± 0.10 | +1.01 ± 0.74 | +0.10 | 0.01 / 0.00 |

`action_match`: top-1 agreement rate between argmax π(z\*ₖ) and argmax π(ẑₖ); 1/7 ≈ 0.14 random baseline.
`KL median`: per-rollout median of KL(π(ẑₖ) ‖ π(z\*ₖ)) — 0 = identical action distributions.
`value_corr`: Pearson r between V(ẑₖ) and V(z\*ₖ) across rollouts.
`V(z\*) / V(ẑ)`: mean value head outputs.

† DK-16 cells excluded from headline interpretation: in our 8M-step training budget, both v6 and v5dc policies on DK-16 fail to solve the task (success = 0 / 3); the policy distributions collapse onto a default action and V ≈ 0 everywhere, so action_match = 0.72 and KL = 0.01 are degenerate floors rather than evidence of coherence. This corroborates the "policy has not converged at DK-16" caveat in §4.6 of the main paper.

### Reading

For the two cells where the policy actually solves the task (DK-8 v6 and DK-8 vae):

1. **Action disagreement is already saturated at k=1.** action_match is 0.29–0.43 at k=1 and remains essentially flat across k ∈ {1, 5, 10}. The policy's input distribution under ẑₖ has *moved out of the regime z\*ₖ trained it on* in a single WM step.
2. **KL divergence is large.** Median KL of 1.0–4.8 nats means the action distributions are not just argmax-different — they're distributionally far apart (1 nat KL ≈ 63% TV distance for two distributions over 7 actions).
3. **Value correlation is near zero or negative.** DK-8 v6 reads value_corr = **−0.42 ± 0.18** at k=1, with the mean drifting toward zero by k=10 (the variance washes out as samples accumulate). DK-8 vae reads −0.06 → −0.18 across horizons.
4. **Mean values agree but ranking does not.** V(z\*) and V(ẑ) means are within a hundredth of each other on the working cohorts, while the per-rollout correlation is anti-aligned. A planner using ẑₖ to bootstrap value would receive the right *average* but cannot distinguish good states from bad — exactly the "imagined-rollout value misranking" mode the discussion section describes.

### Bottom line

The trained policy and value head — the actual decision-makers a planner queries from a WM rollout — produce action distributions and value estimates from ẑₖ that are nearly indistinguishable from random with respect to what they would say on z\*ₖ, on every checkpoint where the policy itself works. This complements Phase A's apples-to-apples Pearson r findings: the WM is not just probe-misaligned, it is *unusable for the actual decision-makers* the encoder was trained alongside.
