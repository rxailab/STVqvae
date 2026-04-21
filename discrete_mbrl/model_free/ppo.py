import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F


def focal_cross_entropy(logits, targets, weight=None, gamma=2.0, reduction='mean'):
    """Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t).
    Down-weights easy (high-confidence correct) examples so the model
    focuses gradient budget on hard/rare cases.

    Args:
        logits: (N, C) unnormalized class scores
        targets: (N,) integer class labels
        weight: (C,) per-class weight tensor (optional, like CE class weights)
        gamma: focusing parameter (0 = standard CE, 2 = strong focusing)
        reduction: 'mean' or 'sum'
    """
    log_p = F.log_softmax(logits, dim=-1)          # (N, C)
    p = log_p.exp()                                 # (N, C)
    # Gather the probabilities of the true class
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (N,)
    log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
    focal_weight = (1 - p_t) ** gamma               # (N,)
    loss = -focal_weight * log_p_t                   # (N,)
    if weight is not None:
        alpha_t = weight[targets]                    # (N,)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    return loss.sum()


def ortho_init(ae, policy, critic):
  """
  Initialize the weights of the actor-critic and autoencoder
  models using orthogonal initialization
  """
  # Unwrap torch.compile's OptimizedModule to allow subscript access
  raw_policy = getattr(policy, '_orig_mod', policy)
  raw_critic = getattr(critic, '_orig_mod', critic)

  for m in ae.modules():
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
      torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)

  for m in raw_policy.modules():
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
      # Check if last layer
      if m == raw_policy[-1]:
        torch.nn.init.orthogonal_(m.weight, gain=0.01)
      else:
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)

  for m in raw_critic.modules():
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
      # Check if last layer
      if m == raw_critic[-1]:
        torch.nn.init.orthogonal_(m.weight, gain=1)
      else:
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)


class PPOTrainer():
  def __init__(
      self, env, policy, critic, ae, optimizer,
      epsilon=1e-7, ppo_iters=20, ppo_clip=0.2, value_coef=0.5,
      minibatch_size=32, entropy_coef=0.003, gae_lambda=0,
      norm_advantages=False, max_grad_norm=0.5, e2e_loss=False,
      target_ae=None, trans_model=None, wm_aux_coef=0.0,
      sem_head=None, sem_aux_coef=0.0, sem_aux_start_reward=0.0, sem_use_class_weights=True,
      sem_focal_gamma=0.0, sem_pre_vq=False, sem_class_weight_power=1.0,
    ):
    self.n_acts = env.action_space.n
    self.policy = policy
    self.critic = critic
    self.ae = ae
    self.target_ae = target_ae  # stable EMA encoder for bootstrapping; None = use self.ae
    self.optimizer = optimizer
    self.device = next(self.policy.parameters()).device
    self.epsilon = epsilon
    self.ppo_iters = ppo_iters
    self.ppo_clip = ppo_clip
    self.value_coef = value_coef
    self.entropy_coef = entropy_coef
    self.minibatch_size = minibatch_size
    self.gae_lambda = gae_lambda
    self.norm_advantages = norm_advantages
    self.max_grad_norm = max_grad_norm
    self.e2e_loss = e2e_loss
    self.trans_model = trans_model      # DiscreteTransitionModel for auxiliary loss
    self.wm_aux_coef = wm_aux_coef     # coefficient for world-model auxiliary loss
    self.sem_head = sem_head            # SemanticHead for per-position object-type prediction
    self.sem_aux_coef = sem_aux_coef   # coefficient for semantic auxiliary loss
    self.sem_aux_start_reward = sem_aux_start_reward  # reward gate: only apply sem_aux after this
    self.sem_aux_active = (sem_aux_start_reward <= 0)  # active immediately if no gate
    self.sem_use_class_weights = sem_use_class_weights  # use inverse-freq weighting
    self.sem_focal_gamma = sem_focal_gamma  # focal loss gamma (0 = standard CE)
    self.sem_pre_vq = sem_pre_vq        # apply sem loss to pre-VQ encoder output (direct grad)
    self.sem_class_weight_power = sem_class_weight_power  # 1.0=inv-freq, 0.5=sqrt, 0=none
    self.sem_class_weights = None       # computed lazily from first batch
    self.policy_losses = []
    self.critic_losses = []
    self.step_idx = 1

  
  def maybe_activate_sem_aux(self, current_reward):
    """Activate semantic aux loss once reward exceeds the start gate."""
    if not self.sem_aux_active and self.sem_head is not None \
            and current_reward >= self.sem_aux_start_reward:
      self.sem_aux_active = True
      print(f'[SEM_AUX] Activated at reward={current_reward:.4f} '
            f'(gate={self.sem_aux_start_reward:.4f})')

  def calculate_gaes(self, rewards, values, next_values, gammas, decay=0.95):
    """
    Return the General Advantage Estimates from the 
    given reward and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    deltas = [rewards[i] + gammas[i] * next_values[i] - values[i] \
      for i in range(len(rewards))]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
      gaes.append(deltas[i] + decay * gammas[i] * gaes[-1])

    return torch.tensor(gaes[::-1])

  def train(self, batch_data):
    """
    Update the policy and critic models using the PPO algorithm

    Args:
      batch_data: A dict of tensors containing the following data:
        obs, states, act_log_probs, next_obs, rewards, acts, gammas
    """
    self.policy.train()

    # Bootstrap rewards if episode is not done
    _stable_ae = self.target_ae if self.target_ae is not None else self.ae
    if batch_data['gammas'][-1] > 0:
      with torch.no_grad():
        next_state = _stable_ae.encode(
          batch_data['next_obs'][-1:].to(self.device),
          return_one_hot=True)
        next_value = self.critic(next_state).squeeze()
      next_value = next_value.cpu()
      batch_data['rewards'][-1] += batch_data['gammas'][-1] * next_value
      batch_data['gammas'][-1] = 0

    # Calculate returns
    returns = torch.zeros(len(batch_data['rewards']))
    returns[-1] = batch_data['rewards'][-1]
    for i in reversed(range(len(batch_data['rewards']) - 1)):
      returns[i] = batch_data['rewards'][i] + \
        batch_data['gammas'][i] * returns[i+1]
      
    obs = batch_data['obs'].to(self.device)
    states = batch_data['states'].to(self.device)
    acts = batch_data['acts'].to(self.device)
    returns = returns.to(self.device)

    with torch.no_grad():
      values = self.critic(states).squeeze(1)
      logits = self.policy(states)
    probs = F.softmax(logits, dim=-1)
    old_act_probs = probs.gather(1, acts).squeeze(1)

    # Calculate advantages
    
    if self.gae_lambda == 0:
      with torch.no_grad():
        advantages = returns - values
    else:
      with torch.no_grad():
        last_state = _stable_ae.encode(
          batch_data['next_obs'][-1:].to(self.device),
          return_one_hot=not self.e2e_loss,
          return_quantized=self.e2e_loss)
        last_value = self.critic(last_state).squeeze(dim=0)

      next_values = torch.cat([values[1:], last_value])
      advantages = self.calculate_gaes(
        batch_data['rewards'], values, next_values,
        batch_data['gammas'], decay=self.gae_lambda)
      advantages = advantages.to(self.device)

      returns = advantages + values

    train_data = {
      'obs': obs,
      'states': states,
      'acts': acts,
      'old_act_probs': old_act_probs,
      'old_values': values,
      'advantages': advantages,
      'returns': returns,
    }
    # Include next_obs for world-model auxiliary loss (predict next state)
    if self.trans_model is not None and self.wm_aux_coef > 0:
      train_data['next_obs'] = batch_data['next_obs'].to(self.device)
    # Include semantic_grid for semantic auxiliary loss (predict per-cell object type)
    if self.sem_head is not None and self.sem_aux_coef > 0 and 'semantic_grid' in batch_data:
      train_data['semantic_grid'] = batch_data['semantic_grid'].to(self.device)

    policy_losses = []
    critic_losses = []
    entropy_losses = []

    for _ in range(self.ppo_iters):
      zipped_buffer = list(zip(*train_data.values()))
      np.random.shuffle(zipped_buffer)
      train_buffer = list(zip(*zipped_buffer))
      train_data = {k: torch.stack(v) for k, v in \
        zip(train_data.keys(), train_buffer)}

      # Break the data into mini-batches for the model updates
      for batch_idx in range(int(np.ceil(len(train_data['states']) / self.minibatch_size))):
        minibatch = {k: v[batch_idx * self.minibatch_size: \
          (batch_idx + 1) * self.minibatch_size] \
          for k, v in train_data.items()}
          
        # Calculate new action probabilities and values for the epoch.
        # For VQ-VAE e2e: use return_quantized so straight-through grads reach encoder.
        # For other models: use return_one_hot (VAE/AE use continuous encode anyway).
        if self.e2e_loss:
            is_vqvae = hasattr(self.ae, 'quantizer') and not getattr(self.ae, 'quantized_enc', False)
            if is_vqvae:
                minibatch['states'] = self.ae.encode(minibatch['obs'], return_quantized=True)
            else:
                minibatch['states'] = self.ae.encode(minibatch['obs'], return_one_hot=True)
        new_values = self.critic(minibatch['states'])
        new_act_probs = F.softmax(self.policy(minibatch['states']), dim=-1)
        policy_entropy = Categorical(probs=new_act_probs).entropy()
        new_act_probs = new_act_probs.gather(1, minibatch['acts'])
        new_act_probs = new_act_probs.squeeze(1)
        new_values = new_values.squeeze(1)

        # Calulcate the value loss
        value_loss = F.mse_loss(new_values, minibatch['returns'])

        if self.norm_advantages:
          mb_advantages = (minibatch['advantages'] - minibatch['advantages'].mean()) \
            / (minibatch['advantages'].std() + 1e-8)
        else:
          mb_advantages = minibatch['advantages']

        # Calculate the policy loss
        # print(new_act_probs, minibatch['old_act_probs'])
        policy_ratio = new_act_probs / (minibatch['old_act_probs'] + self.epsilon)
        clipped_policy_ratio = torch.clamp(policy_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
        policy_loss = torch.min(policy_ratio * mb_advantages,
          clipped_policy_ratio * mb_advantages)
        policy_loss = -policy_loss.mean()
        entropy_loss = -policy_entropy.mean()

        total_loss = policy_loss + self.value_coef * value_loss \
          + self.entropy_coef * entropy_loss

        # World-model auxiliary predictive loss: predict next state from current
        # state + action. Gradient flows through encoder via straight-through,
        # shaping the representation to be temporally predictable.
        wm_aux_loss_val = 0.0
        if self.trans_model is not None and self.wm_aux_coef > 0 and self.e2e_loss:
            # Re-encode current obs with STE (gradient-enabled) — reuse the
            # states we already computed for PPO if they are quantized embeddings
            cont_states = minibatch['states']  # (B, flat_dim), has STE grads

            # Get discrete target indices for next_obs (no gradient needed)
            with torch.no_grad():
                next_indices = self.ae.encode(minibatch['next_obs'])  # (B, n_latent)

            # Predict next state from continuous embeddings (gradient flows to encoder)
            pred_logits, pred_reward, _ = self.trans_model.forward_from_continuous(
                cont_states, minibatch['acts'].squeeze(-1), return_logits=True)

            # Cross-entropy: pred_logits (B, n_embeddings, n_latent) vs next_indices (B, n_latent)
            aux_state_loss = F.cross_entropy(pred_logits, next_indices, reduction='mean')

            # Reward prediction (MSE against actual rewards from batch)
            # Note: minibatch['returns'] includes discounted future, use raw rewards
            # if available; otherwise skip reward aux to avoid noisy targets.
            wm_aux_loss = aux_state_loss
            wm_aux_loss_val = wm_aux_loss.item()
            total_loss = total_loss + self.wm_aux_coef * wm_aux_loss

        # Semantic auxiliary loss: predict per-position object type from current state.
        # Gradient flows through STE back to the encoder, nudging spatial codes to be
        # semantically distinct (wall ≠ lava ≠ empty ≠ agent).
        # Gated by sem_aux_start_reward: only activates after reward exceeds threshold
        # to let the policy anchor the representation first.
        sem_aux_loss_val = 0.0
        if self.sem_head is not None and self.sem_aux_coef > 0 and self.e2e_loss \
                and self.sem_aux_active and 'semantic_grid' in minibatch:
            if self.sem_pre_vq:
                # Re-encode raw obs to get pre-VQ continuous features (direct gradient)
                enc_out = self.ae.encoder(minibatch['obs'])    # (B, C, H, W)
                B_enc = enc_out.shape[0]
                n_lat_side = enc_out.shape[2]
                n_lat_sq = n_lat_side * n_lat_side
                sem_input = enc_out.view(B_enc, -1, n_lat_sq).permute(0, 2, 1)  # (B, n_lat, C)
                # Use a simple linear head directly (SemanticHead's forward expects flat)
                sem_input_flat = enc_out.view(B_enc, -1)  # (B, C*H*W)
                sem_logits = self.sem_head(sem_input_flat)  # (B, n_latent, n_classes)
            else:
                cont_states = minibatch['states']       # (B, n_latent * embedding_dim), STE grads
                sem_logits = self.sem_head(cont_states) # (B, n_latent, n_classes)
            sem_targets = minibatch['semantic_grid'].long()  # (B, n_latent)
            B_sem, n_lat, n_cls = sem_logits.shape

            # Lazy-init inverse-frequency class weights (computed once from first batch)
            if self.sem_class_weights is None and self.sem_use_class_weights:
                flat_t = sem_targets.view(-1)
                counts = torch.bincount(flat_t, minlength=n_cls).float().clamp(min=1)
                inv_freq = counts.sum() / (n_cls * counts)
                # Apply power (1.0=full inv-freq, 0.5=sqrt — gentler for very imbalanced data)
                inv_freq = inv_freq ** self.sem_class_weight_power
                # Normalize over PRESENT classes only (count > 50) — absent classes
                # would otherwise inflate the mean and crush all common-class weights.
                present_mask = counts > 50
                if present_mask.any():
                    mean_present = inv_freq[present_mask].mean()
                    inv_freq = inv_freq / mean_present
                # Absent classes get weight 1.0 (neutral); present classes are normalized
                inv_freq[~present_mask] = 1.0
                # Cap to prevent extreme weight on very rare classes
                inv_freq = inv_freq.clamp(max=10.0)
                self.sem_class_weights = inv_freq.to(self.device)
                n_present = present_mask.sum().item()
                print(f'[SEM_AUX] Class weights (power={self.sem_class_weight_power}, '
                      f'{n_present}/{n_cls} classes present):')
                for i, w in enumerate(inv_freq):
                    if counts[i] > 1:
                        print(f'  class {i:2d}: weight={w:.3f}  count={int(counts[i])}')
                absent = [i for i in range(n_cls) if counts[i] <= 50]
                if absent:
                    print(f'  Absent classes (weight=1.0): {absent}')

            if self.sem_focal_gamma > 0:
                sem_loss = focal_cross_entropy(
                    sem_logits.view(B_sem * n_lat, n_cls),
                    sem_targets.view(B_sem * n_lat),
                    weight=self.sem_class_weights,
                    gamma=self.sem_focal_gamma,
                    reduction='mean'
                )
            else:
                sem_loss = F.cross_entropy(
                    sem_logits.view(B_sem * n_lat, n_cls),
                    sem_targets.view(B_sem * n_lat),
                    weight=self.sem_class_weights,
                    reduction='mean'
                )
            sem_aux_loss_val = sem_loss.item()
            total_loss = total_loss + self.sem_aux_coef * sem_loss

        policy_losses.append(policy_loss.item())
        critic_losses.append(value_loss.item())
        entropy_losses.append(entropy_loss.item())

        # Update the model
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    result = {
      'policy_loss': np.mean(policy_losses),
      'critic_loss': np.mean(critic_losses),
      'entropy_loss': np.mean(entropy_losses),
    }
    if self.trans_model is not None and self.wm_aux_coef > 0:
      result['wm_aux_loss'] = wm_aux_loss_val
    if self.sem_head is not None and self.sem_aux_coef > 0:
      result['sem_aux_loss'] = sem_aux_loss_val
    return result