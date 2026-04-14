from collections import defaultdict
import copy
import os
import sys

import numpy as np
import torch
import torch.optim as optim

from discrete_mbrl.env_helpers import preprocess_obs

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from torch.distributions import Categorical
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from data import SizedReplayBuffer as ReplayBuffer
from ppo import ortho_init, PPOTrainer
from data_logging import *
from shared.models import *
from shared.trainers import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from rl_utils import *
from env_helpers import make_vec_env


def save_best_model(ae_model, policy, critic, optimizer, step, args, avg_reward, suffix="best", target_ae=None, sem_head=None, trans_model=None):
    """
    Save a checkpoint. Filename: {run_name}_{suffix}_model.pt if run_name is set,
    else {suffix}_model.pt (backward compatible).
    """
    _raw_ae_save = getattr(ae_model, '_orig_mod', ae_model)
    checkpoint = {
        'step': int(step),
        'avg_reward': float(avg_reward),
        'ae_model_state_dict': _raw_ae_save.state_dict(),
        'target_ae_state_dict': target_ae.state_dict() if target_ae is not None else None,
        'policy_state_dict': policy.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'sem_head_state_dict': sem_head.state_dict() if sem_head is not None else None,
        'trans_model_state_dict': trans_model.state_dict() if trans_model is not None else None,
        'args': vars(args),
        'model_info': {
            'ae_params': sum(p.numel() for p in ae_model.parameters()),
            'policy_params': sum(p.numel() for p in policy.parameters()),
            'critic_params': sum(p.numel() for p in critic.parameters()),
        }
    }

    save_dir = f"./models/{args.env_name}"
    os.makedirs(save_dir, exist_ok=True)

    run_name = getattr(args, 'run_name', None)
    filename = f"{run_name}_{suffix}_model.pt" if run_name else f"{suffix}_model.pt"
    save_path = f"{save_dir}/{filename}"
    torch.save(checkpoint, save_path)
    print(f"[BEST] Saved {suffix} model to: {save_path}  (avg_reward={avg_reward:.4f})")
    return save_path


def _maybe_compile(model, name="model"):
    """Safely apply torch.compile where supported. Skip on Windows to avoid TritonMissing."""
    if sys.platform.startswith("win"):
        print(f"Skipping torch.compile for {name} on Windows")
        return model

    if os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1":
        print(f"torch.compile disabled via TORCHDYNAMO_DISABLE for {name}")
        return model

    try:
        return torch.compile(model)
    except Exception as e:
        print(f"torch.compile skipped for {name}: {e}")
        return model


def _freeze_params(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_params(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def _snapback_check(current_avg, best_avg, threshold, decline_count, patience,
                     ae_model, best_encoder_sd, train_encoder, encoder_scheduler, step,
                     min_reward=0.0):
    """Check if encoder should be snapped back to best state and frozen.

    Returns (decline_count, train_encoder, encoder_scheduler, frozen_flag).
    """
    if best_avg <= 0 or best_encoder_sd is None:
        return 0, train_encoder, encoder_scheduler, False

    # Don't activate until best_avg exceeds min_reward gate
    if best_avg < min_reward:
        return 0, train_encoder, encoder_scheduler, False

    if current_avg < best_avg * threshold:
        decline_count += 1
    else:
        decline_count = 0

    if decline_count >= patience and train_encoder:
        # Restore encoder to best state and freeze
        raw_ae = getattr(ae_model, '_orig_mod', ae_model)
        raw_ae.load_state_dict(best_encoder_sd)
        _freeze_params(ae_model)
        ae_model.eval()
        train_encoder = False
        encoder_scheduler = None
        print(f"\n[SNAPBACK step {step}] Encoder restored to best state and frozen! "
              f"(current_avg={current_avg:.4f}, best_avg={best_avg:.4f}, "
              f"threshold={threshold}, decline_count={decline_count})")
        return 0, train_encoder, encoder_scheduler, True

    return decline_count, train_encoder, encoder_scheduler, False


def _ema_update(online: torch.nn.Module, target: torch.nn.Module, tau: float):
    """Update target encoder weights: target = tau*target + (1-tau)*online."""
    online_raw = getattr(online, '_orig_mod', online)
    with torch.no_grad():
        for p_on, p_tgt in zip(online_raw.parameters(), target.parameters()):
            p_tgt.data.mul_(tau).add_((1.0 - tau) * p_on.data)


def train(args, encoder_model=None):
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_space = env.action_space
    act_dim = act_space.n

    # Sample obs for model construction
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        sample_obs, _ = reset_result
    else:
        sample_obs = reset_result
    sample_obs = preprocess_obs([sample_obs])

    # Load / construct encoder
    if encoder_model is None:
        ae_model, ae_trainer = construct_ae_model(
            sample_obs.shape[1:], args, latent_activation=True, load=args.load
        )
        if ae_trainer is not None:
            ae_trainer.log_freq = -1
    else:
        ae_model = encoder_model
        ae_trainer = None

    ae_model = _maybe_compile(ae_model, "ae_model")

    # EMA target encoder: stable copy used for rollout & PPO value bootstrapping.
    # Gradients still flow through the online ae_model during PPO updates.
    ema_tau = getattr(args, 'encoder_ema_tau', 1.0)
    _raw_ae = getattr(ae_model, '_orig_mod', ae_model)
    target_ae = copy.deepcopy(_raw_ae).to(args.device)
    _freeze_params(target_ae)
    target_ae.eval()
    use_ema = (ema_tau < 1.0)
    if use_ema:
        print(f'EMA target encoder enabled (tau={ema_tau})')
    else:
        print('EMA target encoder disabled (tau=1.0), using online encoder for rollout')

    # Decide whether the encoder will be trained at all in this run
    # - e2e_loss: PPO gradients flow through encoder
    # - ae_recon_loss: AE trainer updates encoder/decoder
    train_encoder = bool(args.e2e_loss) or (bool(getattr(args, "ae_recon_loss", False)) and ae_trainer is not None)

    # IMPORTANT: if not training encoder, keep it frozen AND in eval mode always
    ae_model = ae_model.to(args.device)
    if train_encoder:
        _unfreeze_params(ae_model)
        ae_model.train()
    else:
        _freeze_params(ae_model)
        ae_model.eval()

    print("Loaded encoder")

    update_params(args)

    # For e2e VQ-VAE, policy receives continuous quantized embeddings (straight-through
    # grads can flow back to the encoder). For frozen VQ-VAE, use discrete one-hot input.
    vqvae_e2e = (args.ae_model_type == 'vqvae' and args.e2e_loss)

    mlp_kwargs = {
        'activation': args.rl_activation,
        'discrete_input': (args.ae_model_type == 'vqvae') and (not vqvae_e2e),
    }

    if args.ae_model_type == 'vqvae':
        input_dim = args.embedding_dim * ae_model.n_latent_embeds
        if not vqvae_e2e:
            mlp_kwargs['n_embeds'] = args.codebook_size
            mlp_kwargs['embed_dim'] = args.embedding_dim
    else:
        input_dim = ae_model.latent_dim

    policy = mlp([input_dim] + args.policy_hidden + [act_dim], **mlp_kwargs).to(args.device)
    critic = mlp([input_dim] + args.critic_hidden + [1], **mlp_kwargs).to(args.device)

    policy = _maybe_compile(policy, "policy")
    critic = _maybe_compile(critic, "critic")

    if args.ortho_init:
        ortho_init(ae_model, policy, critic)

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'AE Model Params: {count_params(ae_model)}')
    print(f'Policy Params: {count_params(policy)}')
    print(f'Critic Params: {count_params(critic)}')

    # Optimizer (only include encoder params if e2e_loss)
    rl_params = list(policy.parameters()) + list(critic.parameters())
    if args.e2e_loss:
        encoder_lr = getattr(args, 'encoder_lr', None) or args.learning_rate
        if encoder_lr != args.learning_rate:
            print(f'Using separate encoder LR: {encoder_lr} (policy LR: {args.learning_rate})')
            optimizer = optim.Adam([
                {'params': rl_params, 'lr': args.learning_rate},
                {'params': list(ae_model.parameters()), 'lr': encoder_lr},
            ], eps=1e-5)
        else:
            optimizer = optim.Adam(rl_params + list(ae_model.parameters()),
                                   lr=args.learning_rate, eps=1e-5)
    else:
        optimizer = optim.Adam(rl_params, lr=args.learning_rate, eps=1e-5)

    # Cosine LR annealing for encoder only (decays encoder_lr -> 0 over training).
    # Uses LambdaLR with per-group lambdas so the policy/critic LR stays constant.
    encoder_scheduler = None
    if getattr(args, 'encoder_lr_cosine', False) and args.e2e_loss:
        import math
        total_batches = int(np.ceil(args.mf_steps / args.batch_size))
        groups = optimizer.param_groups
        # Identify the encoder group: the one whose initial lr matches encoder_lr
        # (only meaningful when encoder_lr differs from policy lr)
        enc_indices = {
            i for i, g in enumerate(groups)
            if abs(g['lr'] - encoder_lr) < 1e-12
        } if encoder_lr != args.learning_rate else set(range(len(groups)))
        T = total_batches
        lambdas = [
            (lambda s, _i=i: 0.5 * (1 + math.cos(math.pi * s / T))) if i in enc_indices
            else (lambda s: 1.0)
            for i in range(len(groups))
        ]
        encoder_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        print(f'Encoder LR cosine annealing: {encoder_lr} -> 0 over {total_batches} batches (policy LR fixed)')

    # ── World model (optional: auxiliary predictive loss + online training) ──
    use_world_model = getattr(args, 'use_world_model', False)
    trans_model = None
    trans_trainer = None
    wm_aux_coef = getattr(args, 'wm_aux_coef', 0.0)
    wm_train_freq = getattr(args, 'wm_train_freq', 1)  # train every N batches

    if use_world_model:
        from model_construction import construct_trans_model
        _raw_ae = getattr(ae_model, '_orig_mod', ae_model)
        # Temporarily disable e2e_loss flag: construct_trans_model rejects
        # discrete trans models with e2e_loss=True (that check is for the old
        # continuous-model e2e path). Our e2e_loss is for the PPO encoder, not
        # the transition model itself.
        _saved_e2e = getattr(args, 'e2e_loss', False)
        args.e2e_loss = False
        trans_model, trans_trainer = construct_trans_model(
            _raw_ae, args, act_space, load=False)
        args.e2e_loss = _saved_e2e
        trans_model = trans_model.to(args.device)
        # Separate optimizer for transition model (keeps PPO optimizer untouched)
        trans_optimizer = optim.Adam(trans_model.parameters(), lr=1e-3)
        # Add transition model params to PPO optimizer IF aux loss is used,
        # so that PPO's backward can update them jointly with encoder/policy.
        if wm_aux_coef > 0:
            optimizer.add_param_group({
                'params': list(trans_model.parameters()),
                'lr': args.learning_rate,
            })
        n_trans_params = sum(p.numel() for p in trans_model.parameters())
        print(f'World model enabled: aux_coef={wm_aux_coef}, train_freq={wm_train_freq}')
        print(f'Transition Model Params: {n_trans_params}')

    # ── Semantic auxiliary head (optional: per-position object-type prediction) ──
    use_semantic_aux = getattr(args, 'use_semantic_aux', False)
    sem_head = None
    sem_aux_coef = getattr(args, 'sem_aux_coef', 0.0)

    if use_semantic_aux and args.e2e_loss:
        sem_head_version = getattr(args, 'sem_head_version', 1)
        n_latent = getattr(ae_model, 'n_latent_embeds', 81)
        if sem_head_version == 2:
            from shared.models.transition_models import SemanticHeadV2
            sem_head = SemanticHeadV2(
                n_latent=n_latent,
                embedding_dim=args.embedding_dim,
                n_classes=getattr(args, 'sem_n_classes', 11),
                hidden_dim=getattr(args, 'sem_head_hidden', 128),
            ).to(args.device)
        else:
            from shared.models.transition_models import SemanticHead
            sem_head = SemanticHead(
                n_latent=n_latent,
                embedding_dim=args.embedding_dim,
                n_classes=getattr(args, 'sem_n_classes', 11),
                hidden_dim=getattr(args, 'sem_head_hidden', 64),
            ).to(args.device)
        n_sem_params = sum(p.numel() for p in sem_head.parameters())
        print(f'Semantic aux enabled: coef={sem_aux_coef}, n_latent={n_latent}, '
              f'n_classes={getattr(args, "sem_n_classes", 11)}, params={n_sem_params}')
        # Add to PPO optimizer so gradients update sem_head jointly with encoder
        optimizer.add_param_group({
            'params': list(sem_head.parameters()),
            'lr': args.learning_rate,
        })

    ppo = PPOTrainer(
        env, policy, critic, ae_model, optimizer,
        ppo_iters=args.ppo_iters,
        ppo_clip=args.ppo_clip,
        minibatch_size=args.ppo_batch_size,
        value_coef=args.ppo_value_coef,
        entropy_coef=args.ppo_entropy_coef,
        gae_lambda=args.ppo_gae_lambda,
        norm_advantages=args.ppo_norm_advantages,
        max_grad_norm=args.ppo_max_grad_norm,
        e2e_loss=args.e2e_loss,
        target_ae=target_ae if use_ema else None,
        trans_model=trans_model if use_world_model else None,
        wm_aux_coef=wm_aux_coef,
        sem_head=sem_head,
        sem_aux_coef=sem_aux_coef,
        sem_aux_start_reward=getattr(args, 'sem_aux_start_reward', 0.0),
        sem_use_class_weights=getattr(args, 'sem_class_weights', True),
        sem_focal_gamma=getattr(args, 'sem_focal_gamma', 0.0),
        sem_pre_vq=getattr(args, 'sem_pre_vq', False),
    )

    replay_buffer = ReplayBuffer(args.replay_size) if args.ae_er_train else None

    run_stats = defaultdict(list)
    ep_info = defaultdict(list)

    # Best tracking
    best_avg_reward = float('-inf')
    recent_rewards = []
    reward_window = 10

    # Snapback: save encoder state at each new best, restore+freeze on decline
    use_snapback = getattr(args, 'encoder_snapback', False)
    snapback_threshold = getattr(args, 'snapback_threshold', 0.5)
    snapback_patience = getattr(args, 'snapback_patience', 100)
    snapback_min_reward = getattr(args, 'snapback_min_reward', 0.0)
    snapback_best_encoder_sd = None   # state_dict of encoder at best reward
    snapback_decline_count = 0        # consecutive episodes below threshold
    encoder_frozen_by_snapback = False
    if use_snapback:
        print(f'Encoder snapback enabled: threshold={snapback_threshold}, patience={snapback_patience}, min_reward={snapback_min_reward}')

    # Global episode stats (run_stats is reset at log_freq)
    all_episode_rewards = []
    all_episode_lengths = []

    num_envs = getattr(args, 'num_envs', 1)
    use_vec_env = (num_envs > 1)

    if use_vec_env:
        vec_env = make_vec_env(args.env_name, num_envs, max_steps=args.env_max_steps)
        curr_obs_np, _ = vec_env.reset()
        curr_obs = torch.from_numpy(curr_obs_np).float()
        ep_rewards = [[] for _ in range(num_envs)]
        steps_per_update = args.batch_size // num_envs
        print(f'Vectorized rollout: {num_envs} envs × {steps_per_update} steps = {args.batch_size} transitions/update')
    else:
        # Rollout init (single env)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            curr_obs, _ = reset_result
        else:
            curr_obs = reset_result
        curr_obs = torch.from_numpy(curr_obs).float()
        ep_rewards = []

    n_batches = int(np.ceil(args.mf_steps / args.batch_size))
    step = 0

    for _batch in tqdm(range(n_batches)):
        # During rollout, ALWAYS eval encoder unless training is explicitly needed.
        # (Even if training encoder, rollout is normally eval to avoid BN/dropout noise.)
        ae_model.eval()
        policy.eval()

        batch_data = {k: [] for k in ['obs', 'states', 'next_obs', 'rewards', 'acts', 'gammas']}

        if use_vec_env:
            # ── Vectorized rollout: N envs stepped simultaneously ──
            # Store time-major [T, N, ...] then permute to env-major [N, T, ...]
            # before PPO so GAE is computed over each env's trajectory independently.
            # Cross-env contamination is prevented by bootstrapping each env's last
            # step (setting gamma=0 at env boundaries).
            model_device = next(ae_model.parameters()).device
            rollout_enc = target_ae if use_ema else ae_model

            obs_list, states_list, acts_list = [], [], []
            next_obs_list, rewards_list, gammas_list = [], [], []
            sem_grids_list = []   # semantic labels for curr_obs at each step

            for _ in range(steps_per_update):
                with torch.no_grad():
                    states = rollout_enc.encode(
                        curr_obs.to(model_device),
                        return_one_hot=not vqvae_e2e,
                        return_quantized=vqvae_e2e,
                    )  # (N, latent_dim)
                    acts = Categorical(logits=policy(states)).sample().cpu()  # (N,)

                next_obs_np, rewards, terminated, truncated, _ = vec_env.step(acts.numpy())
                dones = terminated | truncated
                next_obs = torch.from_numpy(next_obs_np).float()

                # Capture semantic grid (object type per cell) for current obs.
                # For MiniGrid: extract from grid.encode() (available any time).
                # For Crafter: use _sem_view() or last_semantic from wrapper.
                if use_semantic_aux:
                    _n_lat_side = int(np.round(np.sqrt(getattr(ae_model, 'n_latent_embeds', 81))))
                    _sem_batch = []
                    _is_crafter = 'crafter' in args.env_name.lower()
                    for _sub_env in vec_env.envs:
                        _ug = _sub_env.unwrapped
                        if _is_crafter:
                            # Crafter: get 64×64 semantic map directly
                            if hasattr(_ug, 'get_semantic'):
                                _sem_map = _ug.get_semantic()
                            elif hasattr(_ug, '_sem_view'):
                                _sem_map = _ug._sem_view()
                            elif hasattr(_ug, 'last_semantic') and _ug.last_semantic is not None:
                                _sem_map = _ug.last_semantic
                            else:
                                _sem_map = np.zeros((_n_lat_side, _n_lat_side), dtype=np.uint8)
                            _ge_rowmajor = _sem_map  # already (H, W) row-major
                        else:
                            # MiniGrid: grid.encode() is (width, height, 3) col-major
                            _ge = _ug.grid.encode()[:, :, 0].copy()
                            _ge[_ug.agent_pos[0], _ug.agent_pos[1]] = OBJECT_TO_IDX['agent']
                            _ge_rowmajor = _ge.T  # (height, width) row-major
                        if _ge_rowmajor.shape != (_n_lat_side, _n_lat_side):
                            from PIL import Image as _PIL_Image
                            _ge_rowmajor = np.array(
                                _PIL_Image.fromarray(_ge_rowmajor.astype(np.uint8)).resize(
                                    (_n_lat_side, _n_lat_side), resample=0),
                                dtype=np.int64)
                        _sem_batch.append(_ge_rowmajor.flatten())
                    sem_grids_list.append(
                        torch.tensor(np.stack(_sem_batch), dtype=torch.long))  # (N, n_latent)

                # Store full N-tensors per timestep (time-major)
                obs_list.append(curr_obs.clone())
                states_list.append(states.cpu())
                acts_list.append(acts.unsqueeze(1))   # [N, 1]
                next_obs_list.append(next_obs.clone())
                rewards_list.append(torch.tensor(rewards, dtype=torch.float32))
                gammas_list.append(torch.tensor(
                    args.gamma * (1.0 - dones.astype(np.float32)), dtype=torch.float32))

                # Episode tracking
                for i in range(num_envs):
                    ep_rewards[i].append(float(rewards[i]))
                    if dones[i]:
                        ep_r = float(np.sum(ep_rewards[i]))
                        ep_l = int(len(ep_rewards[i]))
                        ep_rewards[i] = []

                        all_episode_rewards.append(ep_r)
                        all_episode_lengths.append(ep_l)
                        recent_rewards.append(ep_r)
                        if len(recent_rewards) > reward_window:
                            recent_rewards.pop(0)

                        current_avg_reward = float(np.mean(recent_rewards))
                        if (len(recent_rewards) >= reward_window) and (current_avg_reward > best_avg_reward) and (current_avg_reward > 0.0):
                            best_avg_reward = current_avg_reward
                            print(f"New best average reward: {best_avg_reward:.4f} (over {len(recent_rewards)} episodes)")
                            if use_snapback and train_encoder:
                                _raw = getattr(ae_model, '_orig_mod', ae_model)
                                snapback_best_encoder_sd = {k: v.clone() for k, v in _raw.state_dict().items()}
                            if args.save:
                                save_best_model(ae_model, policy, critic, optimizer, step, args, best_avg_reward, suffix="best", target_ae=target_ae, sem_head=sem_head, trans_model=trans_model)

                        # Snapback check
                        if use_snapback and train_encoder and not encoder_frozen_by_snapback and len(recent_rewards) >= reward_window:
                            snapback_decline_count, train_encoder, encoder_scheduler, encoder_frozen_by_snapback = \
                                _snapback_check(current_avg_reward, best_avg_reward, snapback_threshold,
                                                snapback_decline_count, snapback_patience,
                                                ae_model, snapback_best_encoder_sd, train_encoder, encoder_scheduler, step,
                                                min_reward=snapback_min_reward)

                        # Semantic aux gate: activate once reward is high enough
                        if use_semantic_aux and len(recent_rewards) >= reward_window:
                            ppo.maybe_activate_sem_aux(current_avg_reward)

                        run_stats['ep_length'].append(ep_l)
                        run_stats['ep_reward'].append(ep_r)

                    update_stats(run_stats, {'reward': float(rewards[i])})

                if step > 0 and step % args.log_freq == 0:
                    log_stats(run_stats, step, args)
                    run_stats = defaultdict(list)

                step += num_envs
                curr_obs = next_obs

            # Bootstrap last step for each non-terminal env.
            # This sets gamma=0 at every env boundary so GAE cannot bleed
            # across envs when we flatten to env-major order below.
            last_gammas = gammas_list[-1]  # [N]
            non_term = last_gammas > 0
            if non_term.any():
                with torch.no_grad():
                    _boot_enc = target_ae if use_ema else ae_model
                    last_next = next_obs_list[-1][non_term].to(model_device)
                    boot_states = _boot_enc.encode(
                        last_next,
                        return_one_hot=not vqvae_e2e,
                        return_quantized=vqvae_e2e,
                    )
                    boot_vals = critic(boot_states).squeeze(-1).cpu()  # [n_non_term]
                rewards_list[-1][non_term] += last_gammas[non_term] * boot_vals
                gammas_list[-1][non_term] = 0.0

            # Stack to [T, N, ...] then permute to [N, T, ...] and flatten [N*T, ...]
            # Env-major order: env0's full trajectory, then env1's, etc.
            # GAE in ppo.train() stays within each env's segment because
            # gammas=0 at every env boundary (from bootstrap above).
            T, N = steps_per_update, num_envs
            obs_T       = torch.stack(obs_list)        # [T, N, *obs]
            states_T    = torch.stack(states_list)     # [T, N, latent]
            acts_T      = torch.stack(acts_list)       # [T, N, 1]
            next_obs_T  = torch.stack(next_obs_list)   # [T, N, *obs]
            rewards_T   = torch.stack(rewards_list)    # [T, N]
            gammas_T    = torch.stack(gammas_list)     # [T, N]

            def _to_env_major(x, trailing_dims):
                # [T, N, *trailing] → [N, T, *trailing] → [N*T, *trailing]
                perm = (1, 0) + tuple(range(2, x.dim()))
                return x.permute(*perm).reshape(N * T, *trailing_dims)

            obs_dim = tuple(obs_T.shape[2:])
            state_dim = tuple(states_T.shape[2:])
            batch_data = {
                'obs':      _to_env_major(obs_T,      obs_dim),
                'states':   _to_env_major(states_T,   state_dim),
                'acts':     _to_env_major(acts_T,     (1,)),
                'next_obs': _to_env_major(next_obs_T, obs_dim),
                'rewards':  rewards_T.permute(1, 0).reshape(N * T),
                'gammas':   gammas_T.permute(1, 0).reshape(N * T),
            }
            if use_semantic_aux and sem_grids_list:
                sem_T = torch.stack(sem_grids_list)   # [T, N, n_latent]
                n_lat = sem_T.shape[-1]
                batch_data['semantic_grid'] = sem_T.permute(1, 0, 2).reshape(N * T, n_lat)

        else:
            # ── Single-env rollout (original path) ──
            for _ in range(args.batch_size):
                with torch.no_grad():
                    env_change = (
                        isinstance(args.env_change_freq, int)
                        and args.env_change_freq > 0
                        and (step + 1) % args.env_change_freq == 0
                    )

                    model_device = next(ae_model.parameters()).device
                    rollout_enc = target_ae if use_ema else ae_model
                    state = rollout_enc.encode(
                        curr_obs.unsqueeze(0).to(model_device),
                        return_one_hot=not vqvae_e2e,
                        return_quantized=vqvae_e2e,
                    )

                    act_logits = policy(state)
                    act_dist = Categorical(logits=act_logits)
                    act_tensor = act_dist.sample().cpu()
                    act_int = int(act_tensor.item())

                batch_data['obs'].append(curr_obs)
                batch_data['states'].append(state.squeeze(0))
                batch_data['acts'].append(act_tensor)

                step_result = env.step(act_int)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result

                done = done or env_change
                next_obs = torch.from_numpy(next_obs).float()
                ep_rewards.append(reward)

                batch_data['next_obs'].append(next_obs)
                batch_data['rewards'].append(torch.tensor(reward).float())
                batch_data['gammas'].append(torch.tensor(args.gamma * (1 - done)).float())

                if 'achievements' in info:
                    for k, v in info['achievements'].items():
                        run_stats[f'achievement/{k}'].append(v)
                        ep_info[f'achievement/{k}'].append(v)

                if replay_buffer is not None:
                    replay_buffer.add_step(curr_obs, act_tensor, next_obs, batch_data['rewards'][-1], batch_data['gammas'][-1])

                if done:
                    if replay_buffer is not None:
                        replay_buffer.add_step(next_obs, act_tensor, next_obs, batch_data['rewards'][-1], batch_data['gammas'][-1])

                    if env_change or args.env_change_freq == 'episode':
                        if args.env_change_type == 'random':
                            env.seeds = [np.random.randint(0, 1000000)]
                        elif args.env_change_type == 'next':
                            env.seeds = [env.seeds[0] + 1]
                        else:
                            raise ValueError(f'Invalid env change type: {args.env_change_type}')

                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        curr_obs, _ = reset_result
                    else:
                        curr_obs = reset_result
                    curr_obs = torch.from_numpy(curr_obs).float()

                    episode_reward = float(np.sum(ep_rewards))
                    episode_length = int(len(ep_rewards))
                    all_episode_rewards.append(episode_reward)
                    all_episode_lengths.append(episode_length)
                    recent_rewards.append(episode_reward)
                    if len(recent_rewards) > reward_window:
                        recent_rewards.pop(0)

                    current_avg_reward = float(np.mean(recent_rewards))
                    if (len(recent_rewards) >= reward_window) and (current_avg_reward > best_avg_reward) and (current_avg_reward > 0.0):
                        best_avg_reward = current_avg_reward
                        print(f"New best average reward: {best_avg_reward:.4f} (over {len(recent_rewards)} episodes)")
                        if use_snapback and train_encoder:
                            _raw = getattr(ae_model, '_orig_mod', ae_model)
                            snapback_best_encoder_sd = {k: v.clone() for k, v in _raw.state_dict().items()}
                        if args.save:
                            save_best_model(ae_model, policy, critic, optimizer, step, args, best_avg_reward, suffix="best", target_ae=target_ae, sem_head=sem_head, trans_model=trans_model)

                    # Snapback check
                    if use_snapback and train_encoder and not encoder_frozen_by_snapback and len(recent_rewards) >= reward_window:
                        snapback_decline_count, train_encoder, encoder_scheduler, encoder_frozen_by_snapback = \
                            _snapback_check(current_avg_reward, best_avg_reward, snapback_threshold,
                                            snapback_decline_count, snapback_patience,
                                            ae_model, snapback_best_encoder_sd, train_encoder, encoder_scheduler, step)

                    run_stats['ep_length'].append(episode_length)
                    run_stats['ep_reward'].append(episode_reward)

                    if 'crafter' in args.env_name.lower():
                        achievement_keys = [k for k in ep_info.keys() if 'achievement' in k]
                        percents = np.array([np.mean(ep_info[k]) * 100 for k in achievement_keys])
                        score = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
                        run_stats['achievement/score'].append(score)

                    ep_rewards = []
                    ep_info = defaultdict(list)
                else:
                    curr_obs = next_obs

                update_stats(run_stats, {'reward': reward})

                if step > 0 and step % args.log_freq == 0:
                    if 'crafter' in args.env_name.lower():
                        achievement_keys = [k for k in run_stats.keys() if 'achievement' in k]
                        percents = np.array([np.mean(run_stats[k]) * 100 for k in achievement_keys])
                        score = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
                        run_stats['achievement/score'].append(score)

                    log_stats(run_stats, step, args)
                    run_stats = defaultdict(list)

                    if step % (args.log_freq * args.checkpoint_freq) == 0:
                        recons = sample_recon_imgs(ae_model, batch_data['obs'], env_name=args.env_name)
                        log_images({'img_recon': recons}, args, step=step)

                step += 1

        # === Updates ===
        policy.train()

        # Only switch encoder to train if we are actually training it
        if train_encoder:
            ae_model.train()
        else:
            ae_model.eval()

        ae_model.to(args.device)
        policy.to(args.device)

        # Vec env path: batch_data values are already tensors; single-env: lists of tensors
        if use_vec_env:
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
        else:
            batch_data = {k: torch.stack(v).to(args.device) for k, v in batch_data.items()}

        # PPO updates
        if step >= args.rl_start_step:
            loss_dict = ppo.train(batch_data)
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    run_stats[k].append(v)
                else:
                    run_stats[k].append(v.item())

        # ── Online transition model training (standalone, optional) ──
        # Train the world model with its own optimizer on the same real data PPO just used.
        # Only enabled when --wm_standalone_train is set. Without it, the transition model
        # is still updated via the aux loss gradient inside the PPO backward pass.
        if use_world_model and trans_trainer is not None and step >= args.rl_start_step \
                and getattr(args, 'wm_standalone_train', False):
            if _batch % wm_train_freq == 0:
                _raw_ae_for_trans = getattr(ae_model, '_orig_mod', ae_model)
                with torch.no_grad():
                    # Encode obs / next_obs to get discrete target indices
                    trans_obs = batch_data['obs']
                    trans_next_obs = batch_data['next_obs']
                    trans_acts = batch_data['acts'].squeeze(-1)
                    trans_rewards = batch_data['rewards']
                    trans_dones = (batch_data['gammas'] == 0).float()

                # Train transition model (single-step, n=1)
                trans_batch = [trans_obs, trans_acts, trans_next_obs, trans_rewards, trans_dones]
                trans_model.train()
                trans_loss_dict, _ = trans_trainer.train(trans_batch, n=1)
                trans_model.eval()

                for k, v in trans_loss_dict.items():
                    run_stats[f'wm_{k}'].append(v.item())

        # EMA update target encoder after PPO batch
        if use_ema and train_encoder:
            _ema_update(ae_model, target_ae, ema_tau)

        # Step encoder LR scheduler (once per batch)
        if encoder_scheduler is not None:
            encoder_scheduler.step()

        # Hard-freeze encoder after freeze_encoder_after steps
        freeze_after = getattr(args, 'freeze_encoder_after', -1)
        if train_encoder and freeze_after > 0 and step >= freeze_after:
            _freeze_params(ae_model)
            ae_model.eval()
            train_encoder = False
            encoder_scheduler = None
            print(f"[Step {step}] Encoder frozen.")

        # AE recon updates (ONLY if enabled AND trainer exists)
        if getattr(args, "ae_recon_loss", False) and ae_trainer is not None:
            for _ in range(args.n_ae_updates):
                if args.ae_er_train:
                    sampled = replay_buffer.sample(args.ae_batch_size or args.batch_size)
                    batch_obs = sampled[0]
                    batch_next_obs = sampled[2]
                else:
                    batch_obs = batch_data['obs']
                    batch_next_obs = batch_data['next_obs']

                loss_dict, ae_stats = ae_trainer.train((batch_obs, None, batch_next_obs))
                for k, v in {**loss_dict, **ae_stats}.items():
                    run_stats[k].append(v.item())

        # If encoder is meant to be frozen, ensure it stays frozen (guardrail)
        if not train_encoder:
            _freeze_params(ae_model)
            ae_model.eval()

    # Save final checkpoint
    if args.save:
        if len(all_episode_rewards) > 0:
            final_avg_reward = float(np.mean(all_episode_rewards[-reward_window:]))
            overall_avg_reward = float(np.mean(all_episode_rewards))
            # if best never improved beyond -inf (e.g., no rewards), set to best observed anyway
            best_report = best_avg_reward if best_avg_reward != float('-inf') else 0.0

            print(f"\nTraining Complete!")
            print(f"   Final {reward_window}-episode average: {final_avg_reward:.4f}")
            print(f"   Overall average reward: {overall_avg_reward:.4f}")
            print(f"   Best rolling average: {best_report:.4f}")

            save_best_model(ae_model, policy, critic, optimizer, step, args, final_avg_reward, suffix="final", target_ae=target_ae, sem_head=sem_head, trans_model=trans_model)
        else:
            print("No episodes completed, saving final model anyway")
            save_best_model(ae_model, policy, critic, optimizer, step, args, 0.0, suffix="final", target_ae=target_ae, sem_head=sem_head, trans_model=trans_model)

    return policy, critic


if __name__ == '__main__':
    mf_arg_parser = make_mf_arg_parser()
    args = get_args(mf_arg_parser)

    if hasattr(args.env_change_freq, "isdecimal") and args.env_change_freq.isdecimal():
        args.env_change_freq = int(args.env_change_freq)

    args = init_experiment('discrete-mbrl-model-free', args)

    if args.wandb:
        args.update({'policy_hidden': interpret_layer_sizes(args.policy_hidden)}, allow_val_change=True)
        args.update({'critic_hidden': interpret_layer_sizes(args.critic_hidden)}, allow_val_change=True)
    else:
        args.policy_hidden = interpret_layer_sizes(args.policy_hidden)
        args.critic_hidden = interpret_layer_sizes(args.critic_hidden)

    policy, critic = train(args)
