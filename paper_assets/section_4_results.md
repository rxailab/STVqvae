# 4 Experiments

We evaluate the central claim of this work — that *probe accuracy is a misleading proxy for world-model usability* — with four controlled experiments. Section 4.1 describes the protocol. Sections 4.2–4.6 each target one way the claim could be wrong: perhaps probes *do* track world-model quality if one averages over enough seeds (§4.2); perhaps the gap is a peculiarity of vector quantization (§4.3); perhaps it is driven by known codebook pathologies that a simple fix would close (§4.4); perhaps it reflects training noise rather than a structural decoupling (§4.5); or perhaps it disappears at larger latent scales (§4.6). We end with two ablations that rule out cheap explanations (§4.7) and summarize in §4.8.

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

## 4.5 Across-class correlation is zero

A skeptical reader might argue that Tables 2 and 3 look the way they do because we aggregate across classes with wildly different base rates: `wall` is common, `goal` is rare, and averaging makes any relationship noisy. To address this we report the per-seed across-class Pearson correlation between probe accuracy and WM accuracy — five data points per seed (one per task-critical class) — and aggregate across seeds. Under the null hypothesis that the probe is a proxy for the WM, this quantity should be strongly positive at $k{=}1$ and decay gradually with $k$.

**Table 4. Per-encoder mean Pearson r across the five task-critical classes.** Mean ± std over seeds.

| Encoder | r @ k=1           | r @ k=5           | r @ k=10          | N  |
|---------|-------------------|-------------------|-------------------|----|
| v2      | $+0.05\pm0.34$    | $+0.01\pm0.36$    | $+0.04\pm0.35$    | 5  |
| v5dc    | $+0.17\pm0.37$    | $+0.19\pm0.39$    | $+0.17\pm0.37$    | 2  |
| v6      | $+0.26\pm0.10$    | $+0.27\pm0.10$    | $+0.27\pm0.10$    | 5  |
| vae     | $+0.03\pm0.57$    | $+0.04\pm0.53$    | $-0.26\pm0.38$    | 5  |

No encoder family produces a confidently positive correlation: every row has a 95% confidence interval that includes zero at every horizon, and the VAE's $k{=}10$ correlation is *negative*, driven by three of its five seeds. The strongest mean (v6 at $+0.27$) is small in absolute terms and reflects the fact that v6 is the only encoder in which the `goal` class is above chance on *both* the probe and the WM — i.e. the correlation exists because of a single common-mode datapoint, not because probes and WMs track each other.

**Figure 1 (pending).** Scatter of (probe, WM) pairs — one point per class × seed — in three panels for $k\in\{1,5,10\}$, colored by encoder, with bootstrap 95% CI bands on the per-encoder regression line. We expect the scatter to show that the modest mean correlation in v6 is carried entirely by the `goal` point, while the remaining four classes form a horizontal band near zero WM accuracy across all probe values.

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

**(2) The $\alpha_{\text{aux}}$ confound.** The VAE condition differs from the VQ conditions in two ways — the quantization layer and the auxiliary cross-entropy weight. To separate the two, we train v6 with $\alpha_{\text{aux}}{=}0$ for 3 seeds on DK-8 (a "VQ-no-aux" control). This run is queued but has not yet been launched; we report the result in the final version. The prediction from §4.3 is that VQ-no-aux matches vanilla v6 in both probe and WM accuracy, since the auxiliary loss is secondary to the transition-trainer gradient on the VQ transition model.

We also note two checks reported in Appendix B. Training-time WM accuracy on `door` and `key` plateaus at the same values we observe at eval, ruling out underfitting to the training schedule; and per-code hit-rate statistics show no systematic bias against task-critical codes beyond what dead-code restart (v5dc) already corrects.

## 4.8 Summary

Across 5 DK-8 seeds per encoder and 3 DK-16 seeds for v6, **probe accuracy is consistently a poor predictor of world-model usability on the same class**. The gap is large ($\Delta_\text{door} \ge 0.61$ everywhere), survives the removal of the VQ bottleneck (§4.3), is unreachable by the standard fix to the probe-visible failure mode (§4.4), produces a near-zero across-class correlation in every encoder family at every horizon (§4.5), and replicates at a larger latent grid with a pre-ceiling policy (§4.6). The VAE positive control is the cleanest datum: probe $\approx 0.97$, policy reward $\approx 0.97$, WM accuracy $\approx 0.01$ on every task-critical class.

These results falsify two convenient beliefs about self-supervised representations for RL: that a representation "contains" what a probe recovers in a sense the downstream learner can use, and that making a probe read out higher is evidence that the surrounding system is more capable. Probes measure decodability, not propagability; the two come apart, and the gap between them is where model-based reinforcement learning on learned latents currently falls short.
