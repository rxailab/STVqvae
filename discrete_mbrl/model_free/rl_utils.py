import argparse

import numpy as np

from data_logging import *


def make_mf_arg_parser():
  # These are the dyna-specific arguments only
  parser = argparse.ArgumentParser()
  parser.add_argument('--policy_hidden', nargs='*', default=[256, 256])
  parser.add_argument('--critic_hidden', nargs='*', default=[256, 256])
  parser.add_argument('--mf_steps', type=int, default=100_000)
  parser.add_argument('--epsilon', type=float, default=0.15)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--n_ae_updates', type=int, default=1)
  parser.add_argument('--ae_batch_size', type=int, default=None)
  parser.add_argument('--replay_size', type=int, default=100_000)
  parser.add_argument('--env_change_freq', default='-1') # 'episode'
  parser.add_argument('--env_change_type', default='random', choices=['random', 'next'])
  parser.add_argument('--rl_start_step', type=int, default=0)
  parser.add_argument('--rl_activation', default='relu', choices=['relu', 'crelu', 'tanh'])
  parser.add_argument('--ortho_init', action='store_true')

  # PPO arguments
  parser.add_argument('--ppo_batch_size', type=int, default=32)
  parser.add_argument('--ppo_iters', type=int, default=20)
  parser.add_argument('--ppo_clip', type=float, default=0.2)
  parser.add_argument('--ppo_value_coef', type=float, default=0.5)
  parser.add_argument('--ppo_entropy_coef', type=float, default=0.003)
  parser.add_argument('--ppo_gae_lambda', type=float, default=0) # 0.95 is recommended if using GAEs
  parser.add_argument('--ppo_norm_advantages', action='store_true')
  parser.add_argument('--ppo_max_grad_norm', type=float, default=0) # 0.5 recommended if used


  # Binary arguments
  parser.add_argument('--ae_recon_loss', action='store_true')
  parser.add_argument('--ae_recon_loss_binary', type=int, default=None)

  # Whether to sample from the ER to train the AE
  parser.add_argument('--ae_er_train', action='store_true')
  parser.add_argument('--ae_er_train_binary', type=int, default=None)

  parser.add_argument('--e2e_loss_binary', type=int, default=None)
  parser.add_argument('--encoder_lr', type=float, default=None,
                      help='Separate LR for encoder during e2e training. '
                           'Defaults to --learning_rate if not set.')
  parser.add_argument('--encoder_lr_cosine', action='store_true', default=False,
                      help='Cosine anneal encoder LR from encoder_lr to 0 over mf_steps.')
  parser.add_argument('--freeze_encoder_after', type=int, default=-1,
                      help='Freeze encoder after this many env steps (-1 = never).')
  parser.add_argument('--num_envs', type=int, default=1,
                      help='Number of parallel environments for rollout collection. '
                           '>1 enables vectorized envs for better GPU utilization.')
  parser.add_argument('--encoder_ema_tau', type=float, default=1.0,
                      help='EMA decay for target encoder (e.g. 0.995). '
                           '1.0 = disabled (no target encoder). '
                           'When < 1.0, rollout uses stable target encoder; '
                           'PPO gradients still flow through online encoder.')
  parser.add_argument('--encoder_snapback', action='store_true', default=False,
                      help='Enable adaptive encoder snapback: save encoder weights at each '
                           'new best reward, restore & freeze when performance declines.')
  parser.add_argument('--snapback_threshold', type=float, default=0.5,
                      help='Freeze encoder when rolling avg drops below best * threshold '
                           '(default 0.5 = 50%% of peak).')
  parser.add_argument('--snapback_patience', type=int, default=100,
                      help='Number of consecutive declining episodes before triggering '
                           'snapback (default 100).')
  parser.add_argument('--snapback_min_reward', type=float, default=0.0,
                      help='Minimum best rolling avg before snapback monitoring activates. '
                           'Prevents premature freezing during early noisy training. '
                           '(default 0.0 = no gate).')
  parser.add_argument('--run_name', type=str, default=None,
                      help='Prefix for saved model filenames, e.g. "e2ephased" saves '
                           'e2ephased_best_model.pt / e2ephased_final_model.pt.')

  # World model integration (auxiliary predictive loss + online training)
  parser.add_argument('--use_world_model', action='store_true', default=False,
                      help='Enable online world model training and auxiliary predictive loss.')
  parser.add_argument('--wm_aux_coef', type=float, default=0.1,
                      help='Coefficient for world-model auxiliary loss (next-state prediction). '
                           'Shapes encoder to learn temporally-predictable representations.')
  parser.add_argument('--wm_train_freq', type=int, default=1,
                      help='Train transition model every N PPO batches.')
  parser.add_argument('--wm_standalone_train', action='store_true', default=False,
                      help='Also train the transition model with its own optimizer after the PPO '
                           'update (in addition to the aux loss in PPO). Exp #30 used this; '
                           'exp #31 omits it to avoid redundant updates.')

  # Semantic auxiliary loss (per-position object-type prediction)
  parser.add_argument('--use_semantic_aux', action='store_true', default=False,
                      help='Enable semantic auxiliary loss: predict MiniGrid object type '
                           'at each spatial token position from the quantized embeddings.')
  parser.add_argument('--sem_aux_coef', type=float, default=0.05,
                      help='Coefficient for semantic auxiliary loss (default 0.05).')
  parser.add_argument('--sem_head_hidden', type=int, default=64,
                      help='Hidden dimension for the SemanticHead MLP (default 64).')
  parser.add_argument('--sem_n_classes', type=int, default=11,
                      help='Number of object-type classes (default 11 = MiniGrid OBJECT_TO_IDX).')
  parser.add_argument('--sem_aux_start_reward', type=float, default=0.0,
                      help='Reward gate for semantic aux: only activate after rolling avg reward '
                           'exceeds this threshold. Prevents sem gradient from destabilising the '
                           'encoder before the policy anchors the representation. (default 0.0 = no gate)')
  parser.add_argument('--sem_class_weights', action='store_true', default=True,
                      help='Use inverse-frequency class weights in semantic CE loss (default True). '
                           'Upweights rare classes (lava, goal, agent) by up to 20×.')
  parser.add_argument('--no_sem_class_weights', action='store_false', dest='sem_class_weights',
                      help='Disable class weighting for semantic CE loss.')
  parser.add_argument('--sem_head_version', type=int, default=1, choices=[1, 2],
                      help='SemanticHead version: 1=original MLP, 2=pos encoding + local conv + deeper MLP')
  parser.add_argument('--sem_focal_gamma', type=float, default=0.0,
                      help='Focal loss gamma for semantic CE. 0=standard CE, 2=strong focusing on hard examples.')
  parser.add_argument('--sem_pre_vq', action='store_true', default=False,
                      help='Apply semantic loss to PRE-VQ encoder output (direct gradient, no STE). '
                           'Bypasses the quantization bottleneck so the encoder gets clean gradient '
                           'to separate visually similar classes (e.g., goal vs empty).')

  parser.set_defaults(ae_recon_loss=False, ppo_norm_advantages=False, ortho_init=False,
                      encoder_lr_cosine=False, use_world_model=False, wm_standalone_train=False,
                      use_semantic_aux=False)
 
  return parser

def interpret_layer_sizes(sizes):
  if isinstance(sizes, (list, tuple)):
    if len(sizes) == 1:
      sizes = sizes[0]
    elif isinstance(sizes[0], int):
      return sizes
    elif isinstance(sizes[0], str):
      return [int(s) for s in sizes]

  if isinstance(sizes, (int, float)):
    return [int(sizes)]
  elif isinstance(sizes, str):
    # Check if single number or list of numbers
    return eval(sizes)
  else:
    raise ValueError(f'Invalid layer sizes format: {sizes}')

def epsilon_greedy_sample(model, obs, epsilon):
  """Samples an action from the model with epsilon-greedy exploration."""
  if np.random.rand() < epsilon:
    return np.random.randint(model.n_acts)
  else:
    return model.predict(obs)

def update_stats(stats, update_dict):
  for k, v in update_dict.items():
    stats[k].append(v)

def log_stats(stats, step, args):
  mean_stats = {k: np.mean(v) for k, v in stats.items()}

  # Create a pretty log string
  log_str = f'\n--- Step {step} ---\n'
  for i, (k, v) in enumerate(mean_stats.items()):
    log_str += f'{k}: {v:.3f}'
    if i < len(mean_stats) - 1:
      if i % 3 == 2:
        log_str += '\n'
      else:
        log_str += '  \t| '
  # print(log_str)

  # Remove nans for Wandb
  mean_stats = {k: v for k, v in mean_stats.items() if not np.isnan(v)}
  mean_stats['step'] = step
  log_metrics(mean_stats, args, step=step)
  
def to_device(tensors, device):
  return [t.to(device) for t in tensors]