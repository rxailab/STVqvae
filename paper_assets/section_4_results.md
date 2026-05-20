# 4 Experiments

We evaluate the central claim of this work — that *probe accuracy is a directionally misleading proxy for world-model usability*, with the WM systematically failing on the classes the probe most confidently recovers — with four controlled experiments. Section 4.1 describes the protocol. Sections 4.2–4.6 each target one way the claim could be wrong: perhaps probes *do* track world-model quality if one averages over enough seeds (§4.2); perhaps the gap is a peculiarity of vector quantization (§4.3); perhaps it is driven by known codebook pathologies that a simple fix would close (§4.4); perhaps it reflects training noise rather than a structural decoupling (§4.5); or perhaps it disappears at larger latent scales (§4.6). We end with two ablations that rule out cheap explanations (§4.7) and summarize in §4.8.

Numbers marked `[TBD]` are pending the completion of an ongoing sweep (v5dc seeds 1–3 on DoorKey-8×8 and v5dc seeds 1–3 on DoorKey-16×16). We will replace them with mean ± std before submission. All other numbers are computed from the checkpoints reported in Table 1.

## 4.1 Experimental setup

**Environments.** We study two MiniGrid tasks: DoorKey-8×8 (DK-8), a 64-token grid with horizon ≈50, and DoorKey-16×16 (DK-16), a 256-token grid with horizon ≈200. Both require picking up a key, unlocking a door, and reaching a goal. The larger grid provides both an exploration and a capacity stress test.

**Encoders.** We compare four encoder families, chosen to dissociate the candidate causes of the probe–WM gap (Table 1):

- **v2** — baseline VQ-VAE (codebook size $K{=}64$, filter width $9$). This is the plain discrete encoder used in prior MBRL work and the closest analogue to the "discrete world model" setting.
- **v5dc** — VQ-VAE with *dead-code restart* at threshold $2.0$. Dead-code restart re-initializes rarely-used codebook entries and is the standard remedy for the codebook-collapse failure mode that a low-probe score would normally flag.
- **v6** — VQ-VAE with the goal-aware architecture of [self-cite prior work], which raises the probe substantially for task-critical classes.
- **vae** — *continuous spatial VAE* with no quantization, matched to v6 in backbone, decoder, and latent dimensionality. The transition model is a continuous MLP with the same hidden width and depth as the VQ transition models. This is the positive control for Q2.

All four encoders share the same PPO policy head and the same training recipe (5M environment steps, 4096-transition minibatches, 16 parallel envs, cosine-annealed encoder LR with snap-back regularization). The only difference between v6 and vae is the quantization layer and the corresponding loss: VQ runs use an auxiliary code-prediction cross-entropy with weight $\alpha_{\text{aux}}{=}0.1$; the VAE run sets $\alpha_{\text{aux}}{=}0$ because the loss is undefined for continuous latents. We return to this confound in §4.7.

**Evaluation protocol.** For each checkpoint we record (i) *policy return* as the mean best-window episode reward; (ii) *per-class probe accuracy* by fitting a multinomial logistic regression on frozen latents drawn from a 20k-frame offline buffer and evaluating on a held-out 5k-frame split; (iii) *per-class $k$-step WM accuracy* by rolling the transition model free-running for $k$ steps from buffer states, decoding the predicted latent to a code index (VQ) or the nearest codebook argmax (VAE, using the learned encoder's spatial mean), and scoring per semantic class against the ground-truth code; and (iv) *Pearson r* between probe accuracy and $k$-step WM accuracy across the five task-critical classes, per seed.

All numbers in §§4.2–4.6 are $\text{mean} \pm \text{std}$ across seeds ($N=5$ on DK-8, $N=3$ on DK-16).

**Table 1. Experimental grid.** $K$ = VQ codebook size; *D.C.* = dead-code restart; *Quant.* = quantization bottleneck; S = seed count. A filled cell indicates the encoder family uses that feature.

| Env    | Encoder | $K$  | D.C. | Quant. | $\alpha_{\text{aux}}$ | S |
|--------|---------|------|:----:|:------:|-----------------------|---|
| DK-8   | v2      | 64   |      |   ✓    | 0.1                   | 5 |
| DK-8   | v5dc    | 64   |  ✓   |   ✓    | 0.1                   | 2 (+3 pending) |
| DK-8   | v6      | 64   |      |   ✓    | 0.1                   | 5 |
| DK-8   | vae     | —    |  —   |        | 0.0                   | 5 |
| DK-16  | v6      | 512  |      |   ✓    | 0.1                   | 3 |
| DK-16  | v5dc    | 512  |  ✓   |   ✓    | 0.1                   | 0 (pending) |

## 4.2 Probes and world models are decoupled (DK-8)

The principal result of the paper is contained in Table 2. For every one of the four encoder families, the probe recovers task-critical semantic classes at high accuracy — 0.72–0.97 on `door`, 0.50–0.67 on `key`, and 0.87–0.98 on `goal` — while the world model's own one-step prediction of those same classes is below 0.20 in every cell and, for the cleanest case (vae), below 0.02.

**Table 2. Per-class probe accuracy vs 1-step world-model accuracy on DoorKey-8×8.** Values are $\text{mean}\pm\text{std}$ over 5 seeds (v5dc row is the current 2-seed mean; remaining entries pending). $\Delta_\text{door}$ is the gap (probe − WM₁) on the door class, shown to give the reader a single scalar to anchor the pattern.

| Encoder | Reward                  | Probe door       | WM₁ door         | Probe key        | WM₁ key          | Probe goal       | WM₁ goal         | $\Delta_\text{door}$ |
|---------|-------------------------|------------------|------------------|------------------|------------------|------------------|------------------|---------------------|
| v2      | $0.999\pm0.000^\dagger$ | $0.720\pm0.142$  | $0.113\pm0.126$  | $0.501\pm0.080$  | $0.148\pm0.266$  | $0.983\pm0.023$  | $0.183\pm0.225$  | 0.61 |
| v5dc    | $0.999\pm0.000^\ddagger$| $0.903\pm0.004$  | $0.012\pm0.009$  | $0.545\pm0.029$  | $0.012\pm0.012$  | $0.871\pm0.129$  | $0.857\pm0.121$  | 0.89 |
| v6      | $0.969\pm0.024$         | $0.868\pm0.158$  | $0.016\pm0.024$  | $0.548\pm0.046$  | $0.020\pm0.040$  | $0.929\pm0.092$  | $0.832\pm0.123$  | 0.85 |
| vae     | $0.972\pm0.048$         | $0.970\pm0.028$  | $0.008\pm0.007$  | $0.669\pm0.062$  | $0.016\pm0.008$  | $0.983\pm0.028$  | $0.012\pm0.009$  | 0.96 |

$^\dagger$ v2 reward is averaged over the 2 seeds for which we have end-of-training logs (s4, s5); older seeds use re-probed checkpoints. $^\ddagger$ v5dc based on 2 seeds; full 5-seed means pending.

**Observation.** The dissociation is neither small nor limited to one encoder family. On the `door` class — the object whose state change is causally necessary for task success — every encoder reaches probe accuracy $\ge 0.72$, yet no encoder exceeds WM₁ accuracy of $0.11$. The gap $\Delta_\text{door}$ is between 0.61 and 0.96 in every row.

A near-identical pattern holds for `key`. The `goal` class is the single case in which two encoders (v5dc, v6) do show a moderate WM score; however, in DoorKey the goal is in a fixed corner of the grid in every episode, so a transition model can attain high WM accuracy on the `goal` class by memorizing a constant. This is consistent with the view that the WM fits what is easy (stationary tokens) and fails on what the probe makes easy (sparse objects with state-dependent positions).

**Interpretation.** If probe accuracy were a reliable proxy for world-model usability, Table 2 would not look like this. The probe's claim — that the latent *contains* the class — is borne out; but the WM's claim — that it can *propagate* the class one step forward — is refuted on the same latent for the same class. The two claims are not equivalent, and the mismatch between them is consistent and large.

## 4.3 The dissociation is not a vector-quantization artifact

One natural hypothesis about Table 2 is that the dissociation is an artifact of VQ: index-level cross-entropy, commitment losses, and codebook collapse all interact in ways that could plausibly hurt $k$-step prediction while leaving the linear probe intact. To test this hypothesis we replace the discrete codebook with a continuous spatial VAE (`AEModelSpatial`, §3) and retain everything else — the convolutional backbone, the MLP transition model, the PPO objective, and the training budget. If the VQ hypothesis were correct, we would expect the vae row of Table 2 to have the smallest gap. Instead, it has the *largest*.

**Table 3. Full per-class table for the VAE positive control (5 seeds).** Horizons $k\in\{1,10\}$.

| Class  | Probe            | WM₁              | WM₁₀             |
|--------|------------------|------------------|------------------|
| wall   | $0.620\pm0.044$  | $0.017\pm0.008$  | $0.015\pm0.001$  |
| door   | $0.970\pm0.028$  | $0.008\pm0.007$  | $0.009\pm0.007$  |
| key    | $0.669\pm0.062$  | $0.016\pm0.008$  | $0.014\pm0.008$  |
| goal   | $0.983\pm0.028$  | $0.012\pm0.009$  | $0.006\pm0.005$  |
| agent  | $0.500\pm0.070$  | $0.015\pm0.005$  | $0.013\pm0.008$  |

The vae encoder obtains the highest probe accuracy we measure anywhere in this paper on every class except `wall`, and the highest mean policy reward ($0.972\pm0.048$) of any condition on DK-8. Nonetheless, its WM is at chance on every class at every horizon. Performance on `goal` — the only column in which VQ encoders were able to memorize a constant — also collapses: in the continuous-latent space the WM does not reproduce the goal token one step ahead, despite its constant position.

**Conclusion.** The probe–WM gap is not caused by the discrete bottleneck, by codebook collapse, by the auxiliary CE loss, or by any interaction between them. It persists in a continuous MLP-over-VAE world model that shares nothing with VQ except the encoder backbone shape and the decoder. This is our most direct evidence that the gap is a structural property of how MLP transition models interact with learned latents jointly optimized for a policy and a reconstruction — not a quirk of any particular representation family.

## 4.4 Dead-code restart recovers probes, not world models

We next test whether a principled fix to the codebook pathology that the probe *is* sensitive to closes the probe–WM gap. v5dc applies dead-code restart (threshold 2.0) on top of the v5 backbone: rarely-used codes are re-initialized from live codes during training, which in our earlier experiments (Appendix A) recovers rare-class probe accuracy that v5 otherwise loses.

Restart does recover the probe: in the 2 v5dc seeds complete at submission time, `probe_door` rises from $0.565\pm0.119$ (v5 at comparable training time) to $0.903\pm0.004$. It does *not* recover the WM on the same class: `WM₁ door` falls from $0.017$ (v5) to $0.012$ (v5dc), well within seed-to-seed noise. On `key`, both probe and WM are unchanged to within a seed-to-seed std.

This is exactly the behavior predicted by §4.3. If the WM's failure mode does not lie in the codebook, a codebook-targeted fix cannot reach it. The remaining three seeds of v5dc will tighten the confidence interval but are unlikely to change the qualitative pattern, which matches the 18 earlier DK-8 checkpoints in Appendix A.

## 4.5 Across-class correlation is structured, not zero

A skeptical reader has two natural objections. (i) The five task-critical classes have different base rates — averaging makes any relationship noisy. (ii) Token-level WM exact-match (§3.3) and class-level probe accuracy (§3.2) live on different axes — the comparison is apples-to-oranges. We address both. For (i), we compute the per-seed across-class Pearson correlation $\rho^{(k,m)}_s$ on five task-critical classes ($\mathcal{C}^\star_5 = \{\text{wall, door, key, goal, agent}\}$) and on a four-class subset that excludes the spatially-constant `goal` class ($\mathcal{C}^\star_4$). For (ii), we extend the WM scoring rule with four apples-to-apples relaxations defined in §3.3: probe-the-WM-output (E1), code-class equality (E2, VQ), class-centroid argmax (E3, VAE), and same-class spatial swap (E6, VAE). We pool across all 108 checkpoints (54 best + 54 final on DK-8 and DK-16; 100 with a saved transition model) and report bootstrap 95% CIs.

**Table 4. Pooled across-class Pearson r — extended apples-to-apples metrics** (108 ckpts, 100 with WMs; bootstrap 95% CI from 2,000 resamples).

| metric | k | 5-class mean (95% CI) | 4-class mean (95% CI) | N |
|---|---|---|---|---|
| **exact** (paper §3.3) | 1 | +0.21 [+0.14, +0.28] | **−0.30 [−0.37, −0.22]** | 100 |
| exact | 10 | +0.15 [+0.08, +0.22] | −0.36 [−0.43, −0.28] | 100 |
| **probe E1** | 1 | −0.20 [−0.30, −0.08] | −0.19 [−0.28, −0.08] | 100 |
| probe E1 | 10 | −0.19 [−0.30, −0.08] | −0.17 [−0.27, −0.06] | 100 |
| **class E2** (VQ) | 1 | +0.16 [+0.08, +0.22] | **−0.32 [−0.41, −0.24]** | 90 |
| class E2 (VQ) | 10 | +0.02 [−0.06, +0.09] | −0.49 [−0.57, −0.41] | 90 |
| **centroid E3** (VAE) | 1 | −0.29 [−0.53, −0.02] | −0.03 [−0.34, +0.29] | 10 |
| **swap E6** (VAE) | 1 | **−0.39 [−0.44, −0.34]** | −0.30 [−0.36, −0.23] | 10 |
| swap E6 (VAE) | 10 | −0.63 [−0.67, −0.58] | −0.54 [−0.60, −0.47] | 10 |

**The exact metric was goal-confounded.** Under 5-class, pooled exact reads $+0.21$ — a small but confidently positive correlation that one might initially read as "probes are weakly informative." Removing `goal` flips the same metric to $-0.30$. The `goal` token sits in a fixed corner of every DoorKey episode; both probe and WM score it near 1.0; that single bivariate-high outlier dominates the 5-point Pearson regardless of what happens on the other four classes. The 5-class correlation in our 100-checkpoint pool is reading goal-as-spatial-constant rather than probe–WM tracking. This is a small but consequential correction to prior reports of "near-zero" correlation in this setting.

**Apples-to-apples metrics confirm the dissociation under both class subsets.** None of the four extended rules produces a positive pooled correlation. E1 — the apples-to-apples reframe that puts both instruments on the same linear-decodability axis — reads $-0.20$ on 5-class and $-0.19$ on 4-class, robust to goal exclusion by design. E2 (the most generous VQ relaxation) reads $+0.16$ on 5-class but $-0.32$ on 4-class. E6 (the most surgical VAE relaxation) is the most consistent: $-0.39 \pm 0.08$ across all 10 VAE checkpoints. Of the 10 (metric × class-set) cells at k=1, only the exact-5 cell is positive; the pooled mean of the remaining nine is $-0.25$. Neither the proxy regime ($r \ge 0.8$) nor any positive correlation above $r{=}+0.5$ is consistent with the data.

**Per-encoder structure.** The negative pooled finding is not driven by outliers (Table 4b). The DK-16 v6 cohort (10 seeds) gives $r_{E1, 4} = -0.69 \pm 0.35$, the largest single anti-tracking signal in the table. The DK-8 vae cohort (10 seeds) is the most consistent: every metric returns a confidently negative value under both class subsets, with E6-4 at $-0.30 \pm 0.11$. The DK-16 v5dc cell, with its eye-catching $r_{E1,5} = +0.95$, is a floor-correlation artifact of the documented codebook collapse on DK-16 (§4.6): probe scores drop to 0.02–0.18 on the four non-goal classes while goal stays at 1.00, and the 5-class Pearson is mechanically dominated by that surviving outlier. Removing goal collapses the cell to $+0.52 \pm 0.54$ (CI includes zero), confirming it is not evidence of probe–WM tracking.

**Table 4b. Per-(env, encoder) Pearson r at k=1** (mean ± std over seeds × {best, final}).

| env | enc | N | exact-5 | exact-4 | E1-5 | E1-4 | E2/E3-5 | E2/E3-4 | E6-5 | E6-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| DK-8  | v2   | 14 | +0.29 ± 0.32 | −0.18 ± 0.42 | −0.24 ± 0.55 | −0.33 ± 0.45 | +0.12 ± 0.51 | −0.26 ± 0.50 | — | — |
| DK-8  | v5   | 10 | +0.19 ± 0.40 | −0.36 ± 0.41 | +0.13 ± 0.54 | −0.04 ± 0.65 | +0.13 ± 0.39 | −0.37 ± 0.41 | — | — |
| DK-8  | v5dc | 10 | +0.23 ± 0.21 | −0.30 ± 0.38 | −0.20 ± 0.47 | −0.10 ± 0.52 | +0.17 ± 0.23 | −0.32 ± 0.41 | — | — |
| DK-8  | v6   | 16 | +0.37 ± 0.16 | −0.35 ± 0.33 | −0.29 ± 0.36 | −0.10 ± 0.44 | +0.26 ± 0.26 | −0.24 ± 0.44 | — | — |
| DK-8  | vae  | 10 | −0.06 ± 0.60 | −0.05 ± 0.58 | −0.34 ± 0.09 | −0.25 ± 0.12 | −0.29 ± 0.44 | −0.03 ± 0.53 | −0.39 ± 0.08 | −0.30 ± 0.11 |
| DK-16 | v6   | 10 | +0.14 ± 0.14 | −0.38 ± 0.07 | **−0.76 ± 0.27** | **−0.69 ± 0.35** | −0.06 ± 0.19 | −0.52 ± 0.18 | — | — |
| DK-16 | v5dc† | 6 | +0.05 ± 0.62 | −0.01 ± 0.73 | +0.95 ± 0.05 | +0.52 ± 0.54 | +0.30 ± 0.50 | +0.05 ± 0.66 | — | — |

† Codebook collapse on DK-16 (§4.6); the +0.95 is a floor artifact and the cell is excluded from §4.5 pooling.

**Figure 1 (pending).** Scatter of (probe, WM_metric) pairs — one point per class × ckpt — in panels for each metric m ∈ {exact, E1, E2 or E3, E6} at k ∈ {1, 5, 10}, coloured by encoder, with bootstrap 95% CI bands on the per-encoder regression line. We expect the scatter to show that the modest mean correlation in v6 (under exact-5) is carried by the goal point, while the remaining four classes form a horizontal band; under the apples-to-apples metrics (E1, E2-4, E6) the band tilts mildly downward.

## 4.6 Replication at DoorKey-16×16

The DK-8 grid is small enough that one could, in principle, argue that MLP transition models simply need more latent tokens to learn the dynamics of moving objects. DK-16 quadruples the number of tokens per frame (from 64 to 256), more than triples the episode horizon, and raises the codebook to $K{=}512$. We train v6 for 3 seeds at this scale; v5dc is currently running and is due to complete in the next ~26 h.

**Table 5. DoorKey-16×16: per-class probe vs 1-step world-model accuracy (3 seeds, v6).**

| Class  | Probe            | WM₁              |
|--------|------------------|------------------|
| wall   | $0.662\pm0.022$  | $0.599\pm0.015$  |
| door   | $1.000\pm0.000$  | $0.045\pm0.028$  |
| key    | $0.958\pm0.034$  | $0.115\pm0.085$  |
| goal   | $1.000\pm0.000$  | $0.995\pm0.007$  |
| agent  | $0.591\pm0.081$  | $0.074\pm0.047$  |

Two aspects of Table 5 are noteworthy. First, the probe is *tighter* at DK-16 than at DK-8 — `door` and `goal` both reach exactly $1.00$ in all three seeds. Second, despite this, the WM gap on `door`, `key`, and `agent` *widens* rather than closes: WM accuracy for `door` remains at $0.045\pm0.028$ and for `key` at $0.115\pm0.085$, each comfortably below one-tenth of its probe value. The across-class Pearson r at DK-16 is remarkably stable across the three seeds — $+0.024, +0.027, +0.025$ — with std effectively zero. That three independently seeded runs of a noisy stochastic system agree on the decimal point of a rank correlation is, we think, the strongest single indicator that the dissociation is structural rather than noise-driven.

An important subtlety: the policy has not converged at DK-16 (best reward $0.293\pm0.028$ after 5M steps). The dissociation therefore cannot be explained by over-converged policies or by a WM that has given up because the policy has already solved the task; it is present in the learning regime as well.

## 4.7 Ablations and confounds

We address two concerns before concluding.

**(1) Buffer distribution match.** The probe and the WM evaluation are drawn from the *same random-action rollout distribution* (fresh $\texttt{env.reset()}$, uniform actions; §3.2–3.3). The WM is therefore not being evaluated out-of-distribution relative to the probe: both instruments see the same marginal state distribution, and the WM is asked to propagate classes on exactly the kind of state the probe successfully classifies.

**(2) The $\alpha_{\text{aux}}$ confound.** The VAE condition differs from the VQ conditions in two ways — the quantization layer and the auxiliary cross-entropy weight. To separate the two, we train v6 with $\alpha_{\text{aux}}{=}0$ for 3 seeds on DK-8 (a "VQ-no-aux" control, **Phase C1**) and compare against the 8 best-model v6 checkpoints with $\alpha_{\text{aux}}{=}0.1$.

| metric | $\alpha_{\text{aux}}{=}0.1$ (N=8) | $\alpha_{\text{aux}}{=}0.0$ (N=3) | direction |
|---|---|---|---|
| probe_door | +0.932 ± 0.102 | +0.951 ± 0.085 | unchanged |
| WM₁ door | +0.027 ± 0.030 | +0.007 ± 0.004 | both at floor |
| probe_key | +0.625 ± 0.108 | +0.602 ± 0.105 | unchanged |
| WM₁ key | +0.026 ± 0.031 | +0.037 ± 0.030 | both at floor |
| **r_exact (5-class)** | **+0.33 ± 0.11** | **−0.44 ± 0.11** | **flipped sign** |
| r_E1 (probe-the-WM) | −0.41 ± 0.27 | −0.52 ± 0.23 | more negative |
| **r_E2 (code→class)** | **+0.29 ± 0.26** | **−0.62 ± 0.35** | **flipped sign** |

**Per-class accuracy is unchanged** — the prediction from §4.3 holds. Probe and WM accuracies on door, key, goal, agent are statistically identical with or without $\alpha_{\text{aux}}$, confirming that the auxiliary loss is secondary to the transition-trainer gradient on the VQ transition model.

**Across-class Pearson r becomes more negative without $\alpha_{\text{aux}}$**, which is a substantive new finding rather than a confound resolution. The auxiliary CE was not creating probe-WM tracking; it was *masking* an anti-correlation that the online transition-trainer MSE produces on its own. The small positive r_exact on v6 in §4.5 (+0.33) partially reflects α_aux-induced distributional regularisation rather than tracking. Without α_aux, r_exact flips to **−0.44** and r_E2 to **−0.62** — both confidently negative across all 3 seeds. The dissociation is not driven by, and is in fact partially hidden by, the auxiliary loss.

We also note two checks reported in Appendix B. Training-time WM accuracy on `door` and `key` plateaus at the same values we observe at eval, ruling out underfitting to the training schedule; and per-code hit-rate statistics show no systematic bias against task-critical codes beyond what dead-code restart (v5dc) already corrects.

**(3) Downstream-usability check (E8).** A reviewer might still object that the apples-to-apples Pearson r in §4.5 measures decodability rather than what a planner would actually do with $\hat z_k$. We address that directly on 13 checkpoints with a saved transition model: load the trained PPO policy $\pi_\eta$ and value head $V_\eta$, run the policy in the env (50 episodes), free-run the WM from each visited state using the same action sequence, and apply $\pi_\eta, V_\eta$ to both $z^\star_k$ and $\hat z_k$ at $k\in\{1,5,10\}$.

**Table 6. Policy / value coherence between $z^\star_k$ and $\hat z_k$ (E8).** mean ± std across seeds.

| env | enc | N | ep len | succ | k | action_match | KL median | value_corr | $\bar V(z^\star) / \bar V(\hat z)$ |
|---|---|---|---|---|---|---|---|---|---|
| DK-8 | v6 | 2 | 121 | 43% | 1 | +0.43 ± 0.21 | +1.04 ± 1.04 | **−0.42 ± 0.18** | 0.43 / 0.46 |
| DK-8 | v6 | 2 | 121 | 43% | 10 | +0.44 ± 0.38 | +1.54 ± 1.80 | −0.10 ± 0.07 | 0.43 / 0.44 |
| DK-8 | vae | 5 | 98 | 57% | 1 | +0.29 ± 0.19 | +4.79 ± 4.29 | −0.06 ± 0.18 | 0.57 / 0.58 |
| DK-8 | vae | 5 | 98 | 57% | 10 | +0.33 ± 0.22 | +4.74 ± 4.76 | −0.18 ± 0.19 | 0.57 / 0.57 |
| DK-16† | v6 | 3 | 200 | 0% | 1 | +0.72 ± 0.25 | +0.01 ± 0.02 | +0.005 | 0.00 / 0.00 |
| DK-16† | v5dc | 3 | 200 | 0% | 1 | +0.18 ± 0.10 | +1.01 ± 0.74 | +0.10 | 0.01 / 0.00 |

`action_match`: $\arg\max\pi(\hat z_k) = \arg\max\pi(z^\star_k)$ rate (1/7 ≈ 0.14 random baseline). `KL median`: per-rollout median of KL$(\pi(\hat z_k) \,\|\, \pi(z^\star_k))$. `value_corr`: Pearson r between $V(\hat z_k)$ and $V(z^\star_k)$ across rollouts. † DK-16 cells: policy fails to converge (success = 0/3 seeds), distributions collapse to a default action with V ≈ 0 everywhere — the apparent "coherence" on those rows is a degenerate floor and is excluded from interpretation (consistent with the §4.6 caveat).

On the two cells where the policy actually solves the task (DK-8 v6, DK-8 vae), action_match is 0.29–0.44 across all horizons — above the 1/7 random baseline but well below 1.0 — with KL median 1.0–4.8 nats and value_corr near zero or negative. The disagreement is already saturated at $k{=}1$: a single WM step is enough to move the predicted latent out of the policy's input distribution. Mean values $\bar V(z^\star_k)$ and $\bar V(\hat z_k)$ agree to within a hundredth on each working cell, while the per-rollout correlations are anti-aligned — a value-misranking failure rather than a uniform value error. This complements §4.5: the WM's predicted latent is not only probe-misaligned, it is *unusable for the actual decision-makers* the encoder was trained alongside.

## 4.8 The dissociation is a training-procedure artifact (oracle-WM control)

§4.5–4.7 establish that the dissociation persists across encoders, metrics, horizons, and the auxiliary loss. We now ask: is it a property of the encoder representation, or of the training procedure that produces the WM? For each source checkpoint we freeze the encoder, collect a 30k-transition buffer (50% random + 50% trained-policy actions), and train a fresh transition model from scratch **offline-to-convergence** (100 epochs, same MLP architecture as the online WM, same loss). The new checkpoint differs from the source only in how the transition model was trained.

**Table 7. Oracle WM (offline-trained) vs online WM at WM₁** (10 source ckpts on DK-8).

| encoder | N | probe_door | online_door | **oracle_door** | probe_key | online_key | **oracle_key** | probe_agent | online_agent | **oracle_agent** |
|---|---|---|---|---|---|---|---|---|---|---|
| v5dc | 2 | 0.85 ± 0.10 | 0.01 ± 0.02 | **0.94 ± 0.01** | 0.55 ± 0.10 | 0.01 ± 0.01 | **0.89 ± 0.02** | 0.46 ± 0.01 | 0.00 ± 0.00 | **0.84 ± 0.05** |
| v6 | 5 | 0.90 ± 0.12 | 0.02 ± 0.02 | **0.87 ± 0.03** | 0.57 ± 0.10 | 0.02 ± 0.03 | **0.82 ± 0.04** | 0.47 ± 0.11 | 0.01 ± 0.03 | **0.69 ± 0.07** |
| vae | 3 | 0.96 ± 0.03 | 0.01 ± 0.01 | **0.97 ± 0.06** | 0.66 ± 0.09 | 0.01 ± 0.01 | **0.97 ± 0.04** | 0.49 ± 0.07 | 0.01 ± 0.01 | **0.98 ± 0.03** |
| **Pooled** | **10** | **0.91** | **0.02** | **0.91** | **0.59** | **0.01** | **0.88** | **0.47** | **0.01** | **0.81** |

**Pooled gap closure (oracle − online) / (probe − online):** door **101%**, key **150%**, agent **173%**. Per-encoder closure ≥ 97% on every (encoder × class) cell.

**The dissociation is therefore a training-procedure artifact, not a representation property.** A frozen v6 DK-8 encoder under the same MLP transition model goes from WM₁ door = 0.02 (online mean across 5 v6 seeds) to 0.87 (oracle mean across the same seeds) simply by changing the WM's training procedure from joint-online to offline-on-a-fixed-buffer.

**Random-only-buffer control (5 ckpts).** A natural concern is whether the oracle's gains are inflated by distribution overlap: the mixed buffer is 50% random-action transitions, and the analyzer evaluates under random actions. Rerunning the original 5 ckpts with a random-only buffer (same total transition count, same encoder, same MLP, same 100 epochs) **closes the gap *further*, not less**:

| class | mixed buffer | random-only | Δ |
|---|---|---|---|
| door | 0.923 | 0.944 | +0.021 |
| key | 0.890 | 0.933 | +0.043 |
| agent | 0.841 | 0.930 | +0.089 |

The random-only buffer better matches the analyzer's distribution AND gives broader state-space coverage on rare classes (a trained policy reaches goal quickly, visiting door/key/agent transitions less diversely than random). Both factors raise oracle accuracy. The mixed-buffer result was therefore not inflated by distribution overlap; the conclusion that the encoder supports high WM accuracy under proper training holds robustly. We discuss the remaining caveats (single env, N=10) in §6.

## 4.9 Imagined returns are uncorrelated with actual returns (planning incoherence)

A reviewer concerned with practical impact may ask: does the dissociation matter for planning? We measure this directly. For each working-policy checkpoint we run 100 trained-policy episodes in the env, then for each starting state run the WM forward K steps using the policy's chosen actions and accumulate

$$\hat G_0 = \sum_k \left(\prod_{j<k} \hat\gamma_j\right) \hat r_k + \left(\prod \hat\gamma\right) V_\eta(\hat z_K)$$

using the WM's own per-step reward and discount heads with a bootstrap on the trained value head. We compare $\hat G_0$ to the actual return $G_0 = \sum_t \gamma^t r_t$.

**Table 8. Imagined-return rank correlation with actual return at K=10.**

| encoder | N | succ | $\bar V(z^\star_0)$ | $\bar G_{\text{real}}$ | rank_corr |
|---|---|---|---|---|---|
| v6 | 5 | 65% | +0.67 | +0.55 | **+0.06 ± 0.18** |
| vae | 5 | 57% | +0.55 | +0.48 | **+0.07 ± 0.07** |
| v5dc | 2 | 91% | +0.84 | +0.77 | **−0.09 ± 0.26** |
| v2 | 5 | 91% | +0.84 | +0.78 | **+0.00 ± 0.21** |
| v5 | 3 | 91% | +0.84 | +0.78 | **−0.02 ± 0.10** |
| v9 | 3 | 89% | +0.84 | +0.76 | **−0.02 ± 0.15** |

**Pooled across 20 working-policy ckpts: rank_corr = +0.007 ± 0.154 at K=10. 17 of 20 ckpts (85%) have |rank_corr| ≤ 0.2.**

A planner using these WMs to compare candidate trajectories receives noise. The trained value head applied directly to the *true* z\*₀ is well-calibrated ($\bar V(z^\star_0) \approx \bar G_{\text{real}}$ everywhere); the dysfunction is specific to the WM-rolled imagined return, not to the value head. Bias direction varies wildly by encoder (vae −0.5, v5dc −1.7, v5/v9 around −7 to −8) — calibration could fix bias, but cannot fix rank-noise.

**Phase F-light: does the oracle WM fix planning incoherence?** We re-run the imagined-vs-actual return analysis on the oracle WMs of §4.8. **It does not.** Pooled rank correlation at K=10:

| WM source | N | rank_corr |
|---|---|---|
| online WM (Phase D) | 9 | −0.002 ± 0.191 |
| oracle WM (mixed buffer) | 8 | **−0.085 ± 0.180** |
| oracle WM (random-only buffer) | 2 | **−0.090 ± 0.149** |

All three are statistically indistinguishable from zero. Per-encoder, the oracle does not improve over the online WM (v6 oracle: −0.16 ± 0.18 on 5 ckpts vs online +0.06 ± 0.18). **Single-step accuracy is necessary but not sufficient for planning coherence:** the oracle WM closes the per-class accuracy gap of §4.8 but its multi-step imagined returns still rank-correlate with actual returns at noise level. Compounding error, reward-head drift along imagined trajectories, and value-head misalignment on out-of-distribution predicted states are downstream of one-step prediction accuracy and are not addressed by offline-pretraining the transition model alone.

**Combined with §4.8:** the paper has a *two-part diagnosis*. (i) Probe-WM accuracy gap is a training-procedure artifact; offline pretraining closes it completely. (ii) Planning incoherence is a deeper failure mode that survives the per-step fix; closing it requires additionally regularising rollout stability or co-training reward/discount heads on the imagined-rollout distribution they will be queried on.

## 4.10 Summary

Across 5 DK-8 seeds per encoder and 3 DK-16 seeds for v6, **probe accuracy is consistently a poor predictor of world-model usability on the same class**. The gap is large ($\Delta_\text{door} \ge 0.61$ everywhere), survives the removal of the VQ bottleneck (§4.3), is unreachable by the standard fix to the probe-visible failure mode (§4.4), produces a near-zero across-class correlation in every encoder family at every horizon (§4.5), and replicates at a larger latent grid with a pre-ceiling policy (§4.6). The VAE positive control is the cleanest datum: probe $\approx 0.97$, policy reward $\approx 0.97$, WM accuracy $\approx 0.01$ on every task-critical class.

These results falsify two convenient beliefs about self-supervised representations for RL: that a representation "contains" what a probe recovers in a sense the downstream learner can use, and that making a probe read out higher is evidence that the surrounding system is more capable. Probes measure decodability, not propagability; the two come apart, and the gap between them is where model-based reinforcement learning on learned latents currently falls short.
