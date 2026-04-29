# 3 Method

This section formalizes the setup evaluated in §4. We first describe the joint encoder–policy–world-model architecture (§3.1), then define the two instruments whose disagreement is the subject of the paper: the *semantic probe* (§3.2) and *$k$-step world-model accuracy* (§3.3). Section 3.4 defines the across-class correlation we report in §4.5, and §3.5 introduces the one architectural novelty required for the positive control — a spatial variant of the continuous VAE that preserves per-token geometry so that per-class WM accuracy is well-defined without quantization.

## 3.1 Setup and architecture

We consider a standard partially observed MDP $\mathcal{M} = (\mathcal{O}, \mathcal{A}, P, r, \gamma)$ with pixel observations $o_t \in \mathcal{O}$, a discrete action set $\mathcal{A}$, unknown transition kernel $P$, scalar reward $r$, and discount $\gamma$. A MiniGrid observation $o_t$ is an $H \times W \times 3$ image in which every cell carries an object class from a fixed vocabulary $\mathcal{C} = \{c_1, \ldots, c_{11}\}$ (empty, wall, door, key, goal, agent, …). We denote the spatial class grid of $o_t$ by $y_t \in \mathcal{C}^{H' \times W'}$, where $H' \times W'$ is the latent grid. This per-cell label is used only for evaluation — it never enters the learning objective.

An agent learns three interacting modules end-to-end:

1. **Encoder.** A convolutional encoder $\phi_\omega: \mathcal{O} \to \mathbb{R}^{D \times H' \times W'}$ maps pixels to a $D$-dimensional spatial feature map. In the VQ setting a product quantizer $q_\psi$ maps each of the $H'W'$ spatial cells to the nearest of $K$ codebook vectors $\{e_k\}_{k=1}^K$, producing a *code grid* $z_t \in \{1, \ldots, K\}^{H' \times W'}$ and a reconstructed feature map $\tilde\phi(o_t)$. In the continuous VAE setting (§3.5) the quantizer is replaced by per-cell mean/variance heads and $z_t$ is taken as the spatial mean; for evaluation we further discretize by nearest-code argmax against a frozen reference codebook.
2. **World model.** A transition model $f_\theta: \mathbb{R}^{D \cdot H'W'} \times \mathcal{A} \to \mathbb{R}^{D \cdot H'W'}$ maps a flattened latent and an action to a predicted next latent. $f_\theta$ is an MLP with hidden width $h$ and depth $L$; the same architecture is used for all encoders. In the discrete setting, $f_\theta$'s output is additionally decoded back to code logits through a linear head; the auxiliary PPO cross-entropy uses this head.
3. **Policy.** A PPO policy $\pi_\eta$ and value function $V_\eta$ consume the flattened latent $\text{vec}(\phi_\omega(o_t))$ with gradients flowing through the encoder (end-to-end PPO).

The joint training objective combines four terms:

$$
\mathcal{L}(\omega, \psi, \theta, \eta) \;=\; \mathcal{L}_{\text{PPO}}(\eta) \;+\; \lambda_{\text{rec}} \,\mathcal{L}_{\text{rec}}(\omega, \psi) \;+\; \lambda_{\text{wm}} \,\mathcal{L}_{\text{wm}}(\omega, \theta) \;+\; \alpha_{\text{aux}} \,\mathcal{L}_{\text{aux}}(\omega, \psi, \eta).
$$

$\mathcal{L}_{\text{rec}}$ is pixel MSE (plus KL for the VAE); $\mathcal{L}_{\text{wm}}$ is the transition trainer's online one-step loss on an FIFO transition buffer — code cross-entropy in the discrete case and feature MSE in the continuous case; $\mathcal{L}_{\text{aux}}$ is a cross-entropy auxiliary on $z_t$ attached to the PPO batch, predicted by $f_\theta$ from $(\text{vec}(\phi_\omega(o_t)), a_t)$ against $z_{t+1}$. This last term is defined only for discrete codes; we therefore set $\alpha_{\text{aux}}{=}0$ for the continuous VAE and discuss the resulting confound in §4.7.

Crucially, all four modules are trained jointly and on-policy. Gradients from the PPO objective flow into the encoder; gradients from the reconstruction objective flow into both encoder and decoder; the transition model is trained on a sliding buffer of actual transitions emitted by the current policy. This is the standard "learned-latent MBRL" training loop as implemented in [cite prior work]. The world model is therefore not trained to convergence on a fixed dataset: it is learning the dynamics of a non-stationary distribution of behavior.

## 3.2 Semantic probes

For every frozen checkpoint we fit a *multinomial logistic probe* over the per-token representation. We collect a buffer $\mathcal{B}_{\text{probe}} = \{(o_t, y_t)\}_{t=1}^{N_{\text{probe}}}$ of $N_{\text{probe}} = 20{,}000$ frames by rolling out a **random-action** policy from a fresh $\texttt{env.reset()}$, resetting on episode termination. The encoder is evaluated in inference mode; each frame contributes $H'W'$ per-token samples $(\phi_\omega(o_t)_{:, i, j},\; y_{t, ij})$ to a single training set.

We fit one shared logistic regression across all token positions,

$$
p_\beta(c \mid \phi_\omega(o)_{:, i, j}) \;\propto\; \exp\!\big(\beta_c^\top \phi_\omega(o)_{:, i, j}\big),
$$

using scikit-learn's `LogisticRegression(class_weight='balanced', C=1.0, max_iter=300)`. The `balanced` weighting sets each class's weight to $N/(|\mathcal{C}| \cdot n_c)$, where $n_c$ is its count in $\mathcal{B}_{\text{probe}}$; this prevents the 11-way classifier from being dominated by `empty` and `wall`.

We report **per-class probe accuracy** as per-class *recall on the training set*,

$$
\text{Probe}_c \;=\; \frac{|\{(i,j,t)\!: y_{t,ij} = c \;\wedge\; \hat{c}_{t,ij} = c\}|}{|\{(i,j,t)\!: y_{t,ij} = c\}|}.
$$

Two properties of this protocol deserve comment, because they push the probe's measured accuracy *upward* and therefore make the dissociation we observe in §4 a conservative estimate.

**(i) Training-set evaluation.** We do not hold out a test split. Linear probes with $O(10^3)$ parameters on $O(10^6)$ samples are far from overfitting in the generalization-gap sense, so the training-set recall is within a few percentage points of a held-out recall on this data; but the reported number is nonetheless an upper bound on probe generalization. If the probe were hand-tuned downward by a held-out split, the gap between Probe$_c$ and WM$_c^{(k)}$ reported in §4 would only widen.

**(ii) Random-action rollout distribution.** The buffer is drawn from random actions, which biases it toward states near episode start (agent exploring near the spawn) and away from states that require key pickup to reach. This makes `door`, `key`, and `goal` instances *rarer* in $\mathcal{B}_{\text{probe}}$ than under the trained policy, but not absent: the 16-env parallel walk at horizon $\approx$50 still exposes doors and keys in every episode. The WM evaluation (§3.3) uses the same random-action distribution, so the probe and the WM are measured on matched states.

Recall (rather than precision or $F_1$) is reported because the dissociation we study concerns *retrieving* task-critical classes — `door`, `key`, `goal` — from the latent, and recall is the most direct per-class measure of this.

## 3.3 $k$-step world-model accuracy

The world-model analogue of the probe is the accuracy with which the transition model can *propagate* class information forward under free-running rollout. We collect $R = 500$ rollouts, each starting from a fresh $\texttt{env.reset()}$ and stepping under **random actions** $a_t \sim \text{Uniform}(\mathcal{A})$ for up to $k_{\max} = 10$ steps (rollouts are truncated at episode end). At each step we record both the real observation $o_t$ produced by the environment and its encoding $z^\star_t = q(\phi_\omega(o_t))$.

For each rollout we roll the transition model *free-running* from the first encoded state:

$$
\hat z_0 = z^\star_0, \qquad
\hat z_{t+1} = \text{decode}\!\big( f_\theta(\text{vec}(\hat z_t), a_t) \big),
$$

where `decode` is argmax over code logits (VQ) or identity (VAE). The WM predicts over the same action sequence that the environment actually received, so $\hat z_k$ and $z^\star_k$ are directly comparable.

**Per-class $k$-step WM accuracy** is

$$
\text{WM}_c^{(k)} \;=\; \frac{\sum_{r, i, j} \mathbb{1}[y^\star_{0, ij, r} = c] \cdot \mathbb{1}[\hat z_{k, ij, r} = z^\star_{k, ij, r}]}{\sum_{r, i, j} \mathbb{1}[y^\star_{0, ij, r} = c]},
$$

where the sum runs over rollouts $r$ and spatial cells $(i, j)$. Three aspects of this definition matter.

**(a) Cells are bucketed by the ground-truth class at $t{=}0$, not at $t{=}k$.** This reads as "of all the cells that started as class $c$, what fraction does the WM still predict correctly $k$ steps later?" — i.e. it measures how well the WM preserves class-$c$ information across the rollout. This choice agrees with the implementation of the code we report (`sem_labels_0` in `analyze_wm_multistep.py:273`). An alternative metric that buckets by class at step $k$ is reasonable too, but conflates the WM's ability to track existing class-$c$ cells with its ability to *generate new* class-$c$ cells (e.g. a door disappearing or an agent teleporting into a position). We avoid that conflation.

**(b) Free-running rollout under random actions.** We feed the WM's own output back as input at every step and use the same random-action policy that generated the probe buffer. This matches the conditions under which a planner would query the WM and uses the same state distribution as the probe, so the two instruments measure representations of matched inputs.

**(c) Discrete agreement, not feature MSE.** For VQ encoders, WM accuracy on cell $(i,j)$ at step $k$ is the exact-match event $\hat z_{k, ij} = z^\star_{k, ij}$ on code indices. For the continuous VAE, where the WM's output is a real-valued vector, we use **cosine-similarity argmax within a frame**: a predicted token at position $(i,j)$ is "correct" iff, among all $H'W'$ tokens in the same frame, the one whose normalized embedding has the highest cosine similarity to $\hat z_{k, ij}$ is the one at position $(i, j)$ in the *true* encoding. This is strictly harder than a continuous MSE criterion and is well-defined without introducing a quantization step that would bias the comparison against VQ.

## 3.4 Probe–WM correlation

To summarize the relationship between the two instruments on a single checkpoint we compute a *per-seed across-class Pearson correlation*:

$$
\rho^{(k)}_s \;=\; \text{corr}\big( \{\text{Probe}_c\}_{c \in \mathcal{C}^{\star}}, \; \{\text{WM}_c^{(k)}\}_{c \in \mathcal{C}^{\star}} \big)_s,
$$

where $\mathcal{C}^{\star} = \{\texttt{wall}, \texttt{door}, \texttt{key}, \texttt{goal}, \texttt{agent}\}$ is the set of five task-critical classes. $\rho^{(k)}_s$ has five data points — one per class — and is therefore a reasonable summary only in aggregate. We report the mean and standard deviation of $\rho^{(k)}_s$ across seeds per encoder in Table 4 of §4.5.

Under the null hypothesis that probe accuracy is a reliable proxy for world-model usability, we would expect $\mathbb{E}_s[\rho^{(1)}_s]$ to be strongly positive ($\gtrsim 0.8$) and to decay gradually with $k$. The observed values (all near zero at $k{=}1$, negative for the VAE at $k{=}10$) motivate the paper's central claim.

## 3.5 AEModelSpatial: a continuous-latent positive control

The main technical barrier to running the positive control of §4.3 is that the standard continuous VAE used in prior MBRL work (`AEModel`) collapses the full $(D, H', W')$ feature map into a single flat $D$-dimensional latent vector. This loses per-cell spatial structure and therefore makes per-class WM accuracy — defined on cells — undefined. To run the control while changing as little else as possible, we introduce `AEModelSpatial`, a drop-in replacement that preserves the latent grid.

**Architecture.** Given the encoder output $\phi_\omega(o) \in \mathbb{R}^{D \times H' \times W'}$, `AEModelSpatial` applies two $1 \times 1$ convolutions:

$$
\mu(o) = \text{Conv}_{1\times 1}^\mu\big(\phi_\omega(o)\big), \qquad
\log\sigma(o) = \text{Conv}_{1\times 1}^\sigma\big(\phi_\omega(o)\big), \qquad \mu, \log\sigma \in \mathbb{R}^{D \times H' \times W'}.
$$

The latent is $z(o) = \mu(o) + \sigma(o) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$; at inference we use $\mu$. For use by the flat MLP transition model and the PPO policy head, $z$ is flattened to $\mathbb{R}^{D \cdot H' \cdot W'}$; the original spatial structure is recovered by reshape whenever per-cell access is needed (probing, WM accuracy).

**Loss.** `AEModelSpatial` is trained under a standard $\beta$-VAE objective

$$
\mathcal{L}_{\text{rec}}(\omega, \psi) \;=\; \mathbb{E}_{z \sim q_\omega}[\|\text{dec}(z) - o\|^2] \;+\; \beta \cdot D_{\text{KL}}\big(q_\omega(z \mid o) \,\|\, \mathcal{N}(0, I)\big),
$$

with $\beta$ matched to the default VQ commitment coefficient. The decoder, policy head, and transition MLP are otherwise identical to v6.

**Per-class WM scoring.** Because the VAE latent is continuous, we cannot compare `$\hat z_k = z^\star_k$` as code indices. The scoring rule defined in §3.3(c) — cosine-similarity argmax within the frame — is applied *directly* to the per-token slices of the flat latent, with no auxiliary codebook. This keeps the evaluation protocol as close as possible to the VQ case (both reduce to "each predicted token must match its true token better than any distractor in the same frame") while avoiding the methodological freedom of hand-choosing a post-hoc quantizer for the continuous model.

All four encoder families (v2, v5dc, v6, vae) are therefore evaluated under the same three instruments — probe recall (§3.2), free-running $k$-step per-token agreement (§3.3), and across-class Pearson $r$ (§3.4) — in every condition of §4.
