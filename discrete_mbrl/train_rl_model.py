"""
Fixed train_rl_model.py with gym/gymnasium compatibility.

The main issue is that PredictedModelWrapper and TimeLimit from gym
are not compatible with stable_baselines3 which expects gymnasium environments.
"""

import os
import sys
import time
import warnings

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import gymnasium
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit as GymnasiumTimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Import your modules
from shared.models import *
from shared.trainers import *
from data_helpers import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from utils import obs_to_img

SB3_DIR = 'mbrl_runs/'
N_EXAMPLE_ROLLOUTS = 4
N_EVAL_EPISODES = 20
EVAL_INTERVAL = 10
EVAL_UNROLL_STEPS = 20


def parse_stage_schedule(args):
    """Return an optional PPO curriculum as [(unroll_steps, train_steps), ...]."""
    stage_unrolls = [s.strip() for s in getattr(args, 'rl_stage_unrolls', '').split(',') if s.strip()]
    stage_steps = [s.strip() for s in getattr(args, 'rl_stage_steps', '').split(',') if s.strip()]

    if not stage_unrolls and not stage_steps:
        return []
    if not stage_unrolls or not stage_steps:
        raise ValueError('Both --rl_stage_unrolls and --rl_stage_steps must be provided together.')
    if len(stage_unrolls) != len(stage_steps):
        raise ValueError('--rl_stage_unrolls and --rl_stage_steps must have the same number of entries.')

    schedule = []
    for idx, (unroll, steps) in enumerate(zip(stage_unrolls, stage_steps), start=1):
        unroll_steps = int(unroll)
        train_steps = int(steps)
        if train_steps <= 0:
            raise ValueError(f'RL stage {idx} has non-positive train steps: {train_steps}')
        schedule.append((unroll_steps, train_steps))
    return schedule


class GymnasiumCompatWrapper(gymnasium.Env):
    """
    Wrapper to convert a gym-style environment to gymnasium-compatible.
    This fixes the compatibility issue with stable_baselines3.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env

        # Copy over spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Handle metadata
        if hasattr(env, 'metadata'):
            self.metadata = env.metadata
        else:
            self.metadata = {'render_modes': []}

    def reset(self, seed=None, options=None):
        """Reset with gymnasium API (returns obs, info)"""
        if seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)

        result = self.env.reset()

        # Handle both old and new API
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        return obs, info

    def step(self, action):
        """Step with gymnasium API (returns obs, reward, terminated, truncated, info)"""
        result = self.env.step(action)

        if len(result) == 5:
            # Already new API
            return result
        elif len(result) == 4:
            # Old API: convert to new
            obs, reward, done, info = result
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
        else:
            raise ValueError(f"Unexpected step result length: {len(result)}")

    def render(self):
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, 'close'):
            return self.env.close()

    def __getattr__(self, name):
        """Delegate attribute access to wrapped env"""
        return getattr(self.env, name)


class PredictedModelWrapperGymnasium(gymnasium.Env):
    """
    Gymnasium-compatible version of PredictedModelWrapper.
    Wraps encoder and transition model to create a world model environment.
    """

    def __init__(self, base_env, encoder, trans_model):
        super().__init__()
        self.base_env = base_env
        self.encoder = encoder
        self.trans_model = trans_model
        self.device = next(self.encoder.parameters()).device

        # Get observation shape from encoder output
        test_input = np.ones(base_env.observation_space.shape)
        test_input = preprocess_obs([test_input])
        with torch.no_grad():
            obs_shape = self.encoder.encode(test_input.to(self.device)).shape[1:]

        # Define spaces
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = base_env.action_space

        self._curr_obs = None
        self.metadata = {'render_modes': []}

    def _preprocess_obs(self, obs):
        """Convert observation to tensor and encode"""
        obs = preprocess_obs([obs])
        with torch.no_grad():
            encoded = self.encoder.encode(obs.to(self.device))[0]
        return encoded.cpu()

    def _preprocess_action(self, action):
        """Convert action to tensor"""
        return preprocess_act([action])[0]

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None and hasattr(self.base_env, 'seed'):
            self.base_env.seed(seed)

        result = self.base_env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self._curr_obs = self._preprocess_obs(obs)
        return self._curr_obs.numpy(), info

    def step(self, action):
        """Take a step using the world model"""
        action_tensor = self._preprocess_action(action)

        with torch.no_grad():
            trans_out = self.trans_model(
                self._curr_obs.unsqueeze(0).to(self.device),
                action_tensor.unsqueeze(0).to(self.device)
            )
            next_obs, reward, gamma = [x[0].cpu() for x in trans_out]

        self._curr_obs = next_obs

        # Convert gamma to termination signal
        # gamma close to 0 means episode should end
        terminated = gamma.item() < 0.5
        truncated = False

        info = {}

        return next_obs.numpy(), reward.item(), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        if hasattr(self.base_env, 'close'):
            return self.base_env.close()


class ObsEncoderWrapperGymnasium(gymnasium.ObservationWrapper):
    """
    Gymnasium-compatible wrapper that encodes observations using the trained encoder.
    """

    def __init__(self, env, encoder):
        # First wrap if needed
        if not isinstance(env, gymnasium.Env):
            env = GymnasiumCompatWrapper(env)

        super().__init__(env)
        self.encoder = encoder
        self.device = next(self.encoder.parameters()).device

        # Get output shape
        test_input = np.ones(env.observation_space.shape)
        test_input = preprocess_obs([test_input])
        with torch.no_grad():
            obs_shape = self.encoder.encode(test_input.to(self.device)).shape[1:]

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        obs_tensor = preprocess_obs([obs])
        with torch.no_grad():
            encoded = self.encoder.encode(obs_tensor.to(self.device))[0]
        return encoded.cpu().numpy()


def traj_to_imgs(obs_buffer, episode_starts, encoder,
                 n_imgs=N_EXAMPLE_ROLLOUTS, wandb_format=True):
    """Convert trajectory observations to images for logging"""
    if isinstance(obs_buffer, np.ndarray):
        obs_buffer = torch.from_numpy(obs_buffer)
    idx = 0
    traj_imgs = []
    device = next(encoder.parameters()).device

    for _ in range(n_imgs):
        if idx >= len(obs_buffer):
            break
        traj_obs = [obs_buffer[idx]]

        for _ in range(EVAL_UNROLL_STEPS - 1):
            idx += 1
            if idx >= len(obs_buffer) or episode_starts[idx]:
                break
            traj_obs.append(obs_buffer[idx])

        traj_obs = torch.stack(traj_obs)
        decoded_obs = encoder.decode(traj_obs.to(device))
        img = obs_to_img(decoded_obs, cat=True)
        if wandb_format:
            import wandb
            img = wandb.Image(img)
        traj_imgs.append(img)
    return traj_imgs


def rl_eval(rl_model, env, encoder, n_eval_episodes=10, log=True):
    """Evaluate the RL model"""
    obs_buffer = [[] for _ in range(env.num_envs)]
    episode_starts = [[] for _ in range(env.num_envs)]

    last_episode_starts = [True] * env.num_envs

    def callback(locals, _):
        nonlocal last_episode_starts
        curr_obs = locals['observations']
        dones = locals['dones']
        for i, o in enumerate(curr_obs):
            obs_buffer[i].append(o)
            episode_starts[i].append(last_episode_starts[i])
            last_episode_starts[i] = dones[i]

    ep_rewards, ep_lengths = evaluate_policy(
        rl_model, env, callback=callback,
        n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    mean_reward = np.mean(ep_rewards)
    reward_std = np.std(ep_rewards)
    mean_len = np.mean(ep_lengths)

    if not log:
        return mean_reward, reward_std, mean_len

    # Flatten observation buffer
    obs_buffer = np.concatenate([np.stack(obs) for obs in obs_buffer], axis=0)
    episode_starts = np.concatenate([np.stack(starts) for starts in episode_starts], axis=0)

    traj_imgs = traj_to_imgs(obs_buffer, episode_starts, encoder)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import wandb
        wandb.log({
            'rl/eval/ep_reward_mean': np.mean(ep_rewards),
            'rl/eval/ep_reward_std': np.std(ep_rewards),
            'rl/eval/ep_mean_len': np.mean(ep_lengths),
            'rl/eval/trajectory': traj_imgs,
        })

    return mean_reward, reward_std, mean_len


def make_world_model_env(args, base_env, encoder_model, trans_model, unroll_steps):
    """Build a monitored world-model environment with an optional time limit."""
    world_model = PredictedModelWrapperGymnasium(base_env, encoder_model, trans_model)
    if unroll_steps > 0:
        world_model = GymnasiumTimeLimit(world_model, max_episode_steps=unroll_steps)
    return Monitor(world_model)


def make_eval_env(args, encoder_model, max_episode_steps=None):
    """Create a real environment that emits encoded observations."""
    def _factory():
        base_env = make_env(args.env_name, max_steps=args.env_max_steps)
        if max_episode_steps and max_episode_steps > 0:
            base_env = GymnasiumTimeLimit(base_env, max_episode_steps=max_episode_steps)
        encoded_env = ObsEncoderWrapperGymnasium(base_env, encoder_model)
        return Monitor(encoded_env)

    return DummyVecEnv([_factory])


def make_dyna_vec_env(args, encoder_model, trans_model):
    """Create a mixed VecEnv with real and imagined environments (Dyna-style).

    Real environments provide grounded reward signal from the actual MDP.
    Imagined environments provide data-augmented experience using the
    transition model, limited to short horizons where open-loop accuracy
    is still reliable.
    """
    env_fns = []

    # Real environments: real dynamics, encoder-wrapped observations
    for i in range(args.n_real_envs):
        def _make_real(idx=i):
            base_env = make_env(args.env_name, max_steps=args.env_max_steps)
            encoded_env = ObsEncoderWrapperGymnasium(base_env, encoder_model)
            return Monitor(encoded_env)
        env_fns.append(_make_real)

    # Imagined environments: transition-model dynamics, short horizon
    for i in range(args.n_imagined_envs):
        def _make_imagined(idx=i):
            base_env = make_env(args.env_name, max_steps=args.env_max_steps)
            wm_env = PredictedModelWrapperGymnasium(base_env, encoder_model, trans_model)
            wm_env = GymnasiumTimeLimit(wm_env, max_episode_steps=args.dyna_horizon)
            return Monitor(wm_env)
        env_fns.append(_make_imagined)

    n_total = args.n_real_envs + args.n_imagined_envs
    print(f'Dyna VecEnv: {args.n_real_envs} real + {args.n_imagined_envs} imagined '
          f'(horizon={args.dyna_horizon}) = {n_total} envs')
    return DummyVecEnv(env_fns)


def make_real_finetune_vec_env(args, encoder_model, n_envs=4):
    """Create a VecEnv of pure real environments for finetuning."""
    env_fns = []
    for i in range(n_envs):
        def _make(idx=i):
            base_env = make_env(args.env_name, max_steps=args.env_max_steps)
            encoded_env = ObsEncoderWrapperGymnasium(base_env, encoder_model)
            return Monitor(encoded_env)
        env_fns.append(_make)
    print(f'Real finetune VecEnv: {n_envs} real envs')
    return DummyVecEnv(env_fns)


def get_rl_checkpoint_dir(args):
    """Store PPO checkpoints inside the experiment model_dir when available."""
    if getattr(args, 'model_dir', None):
        return os.path.join(args.model_dir, 'rl_models')
    return f'./models/{args.env_name}'


def train_rl_model(args, encoder_model=None, trans_model=None):
    """Main training function for RL with world model"""

    if args.wandb:
        global wandb
        import wandb

    # Create base environment
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_dim = env.action_space.n

    # Get sample observation
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        sample_obs, _ = reset_result
    else:
        sample_obs = reset_result
    sample_obs = preprocess_obs([sample_obs])

    # Load encoder
    if encoder_model is None:
        encoder_model = construct_ae_model(sample_obs.shape[1:], args)[0]
    encoder_model = encoder_model.to(args.device)
    freeze_model(encoder_model)
    encoder_model.eval()
    print('Loaded encoder')

    # Load transition model
    if trans_model is None:
        trans_model = construct_trans_model(encoder_model, args, env.action_space)[0]
    trans_model = trans_model.to(args.device)
    freeze_model(trans_model)
    trans_model.eval()
    print('Loaded transition model')

    if args.wandb:
        wandb.config.update(args, allow_val_change=True)

    use_dyna = getattr(args, 'dyna', False)
    finetune_steps = getattr(args, 'rl_finetune_steps', 0)

    stage_schedule = parse_stage_schedule(args)
    initial_unroll = stage_schedule[0][0] if stage_schedule else args.rl_unroll_steps

    # Create PPO model — either Dyna (mixed real+imagined) or pure imagined
    ppo_n_steps = getattr(args, 'ppo_n_steps', 2048)

    if use_dyna:
        training_env = make_dyna_vec_env(args, encoder_model, trans_model)
        model = PPO('MlpPolicy', training_env, verbose=1, device=args.device,
                    n_steps=ppo_n_steps)
    else:
        world_model = make_world_model_env(args, env, encoder_model, trans_model, initial_unroll)
        model = PPO('MlpPolicy', world_model, verbose=1, device=args.device,
                    n_steps=ppo_n_steps)

    # Create evaluation environment (uses real observations encoded)
    # Always apply rl_eval_max_episode_steps so eval doesn't run forever
    eval_max_steps = getattr(args, 'rl_eval_max_episode_steps', 500)
    eval_env = make_eval_env(args, encoder_model, max_episode_steps=eval_max_steps)

    class RLTrainingCallback(BaseCallback):
        def __init__(self, encoder, eval_env=None, best_model_path=None, verbose=0):
            super().__init__(verbose)
            self.encoder = encoder
            self.eval_env = eval_env
            self.best_model_path = best_model_path
            self.last_eval_timestep = 0
            self.best_eval_reward = -np.inf
            self.rollout_idx = 0

        def _on_step(self):
            return True

        def _maybe_log_train_metrics(self, learner):
            if not args.wandb:
                return

            ep_info_buffer = learner.ep_info_buffer
            if len(ep_info_buffer) == 0:
                return

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wandb.log({
                    'rl/train/ep_reward_mean': np.mean([ep['r'] for ep in ep_info_buffer]),
                    'rl/train/ep_mean_len': np.mean([ep['l'] for ep in ep_info_buffer]),
                    'rl/train/step': learner.num_timesteps,
                })

        def _maybe_eval(self, learner):
            if self.eval_env is None:
                return

            eval_freq = max(0, getattr(args, 'rl_eval_freq', 0))
            fallback_eval = args.wandb and EVAL_INTERVAL > 0 and self.rollout_idx % EVAL_INTERVAL == 0
            freq_eval = eval_freq > 0 and (learner.num_timesteps - self.last_eval_timestep) >= eval_freq

            if not fallback_eval and not freq_eval:
                return

            print(f'Running RL Eval at {learner.num_timesteps} timesteps...')
            mean_reward, std_reward, mean_len = rl_eval(
                learner,
                self.eval_env,
                self.encoder,
                n_eval_episodes=getattr(args, 'rl_eval_episodes', N_EVAL_EPISODES),
                log=args.wandb,
            )
            print(f'Eval: reward={mean_reward:.3f}±{std_reward:.3f}, len={mean_len:.2f}')
            self.last_eval_timestep = learner.num_timesteps

            if mean_reward > self.best_eval_reward:
                self.best_eval_reward = mean_reward
                if self.best_model_path is not None:
                    learner.save(self.best_model_path)
                    print(f'Saved new best real-eval PPO checkpoint to {self.best_model_path}')

        def _on_rollout_end(self):
            learner = self.locals['self']
            self._maybe_log_train_metrics(learner)
            self._maybe_eval(learner)
            self.rollout_idx += 1

    save_dir = None
    legacy_save_dir = None
    best_model_path = None
    if args.save:
        save_dir = get_rl_checkpoint_dir(args)
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, 'ppo_world_model_best')
        legacy_save_dir = f"./models/{args.env_name}"
        if os.path.abspath(legacy_save_dir) != os.path.abspath(save_dir):
            os.makedirs(legacy_save_dir, exist_ok=True)

    callback = RLTrainingCallback(encoder_model, eval_env, best_model_path=best_model_path)

    # Train
    try:
        if stage_schedule:
            total_stage_steps = sum(stage_steps for _, stage_steps in stage_schedule)
            print(f'Using RL stage schedule: {stage_schedule}')
            if args.rl_train_steps > 0 and total_stage_steps != args.rl_train_steps:
                print(f'Warning: rl_train_steps={args.rl_train_steps} but staged total={total_stage_steps}; using staged total.')

            for stage_idx, (stage_unroll, stage_steps) in enumerate(stage_schedule, start=1):
                print(f'Starting RL stage {stage_idx}/{len(stage_schedule)}: horizon={stage_unroll}, steps={stage_steps}')
                if stage_idx > 1:
                    env.close()
                    env = make_env(args.env_name, max_steps=args.env_max_steps)
                    world_model = make_world_model_env(args, env, encoder_model, trans_model, stage_unroll)
                    model.set_env(world_model)

                stage_eval_max_steps = getattr(args, 'rl_eval_max_episode_steps', 500)
                stage_eval_env = make_eval_env(args, encoder_model, max_episode_steps=stage_eval_max_steps)
                try:
                    model.learn(stage_steps, callback=callback, reset_num_timesteps=False)
                    matched_reward, matched_std, matched_len = rl_eval(
                        model, stage_eval_env, encoder_model, n_eval_episodes=N_EVAL_EPISODES, log=False
                    )
                    print(
                        f'Stage {stage_idx} matched real-env eval: '
                        f'reward={matched_reward:.3f}±{matched_std:.3f}, len={matched_len:.2f}'
                    )
                finally:
                    stage_eval_env.close()
        else:
            model.learn(args.rl_train_steps, callback=callback)
    except KeyboardInterrupt:
        print('Training interrupted')

    # --- Phase 2: Real-env finetuning (optional) ---
    if finetune_steps > 0:
        print(f'\n=== Phase 2: Real-env finetuning for {finetune_steps} steps ===')
        # set_env requires same n_envs as training; fill with real envs
        n_total = model.n_envs
        finetune_env = make_real_finetune_vec_env(
            args, encoder_model, n_envs=n_total)
        model.set_env(finetune_env)

        # Reset eval tracking for the finetune phase
        finetune_callback = RLTrainingCallback(
            encoder_model, eval_env, best_model_path=best_model_path)
        try:
            model.learn(finetune_steps, callback=finetune_callback, reset_num_timesteps=False)
        except KeyboardInterrupt:
            print('Finetuning interrupted')
        finally:
            finetune_env.close()

    # Final evaluation
    mean_reward, std_reward, mean_len = rl_eval(
        model, eval_env, encoder_model, n_eval_episodes=N_EVAL_EPISODES, log=args.wandb
    )
    print(f'Final: reward={mean_reward:.3f}±{std_reward:.3f}, len={mean_len:.2f}')

    selected_model = model
    selected_label = 'final'

    best_model_zip = f'{best_model_path}.zip' if best_model_path is not None else None
    if best_model_zip and os.path.exists(best_model_zip):
        best_eval_env = make_eval_env(
            args,
            encoder_model,
            max_episode_steps=getattr(args, 'rl_eval_max_episode_steps', 500),
        )
        try:
            best_model = PPO.load(best_model_path, env=best_eval_env, device=args.device)
            best_mean_reward, best_std_reward, best_mean_len = rl_eval(
                best_model,
                best_eval_env,
                encoder_model,
                n_eval_episodes=N_EVAL_EPISODES,
                log=False,
            )
            print(
                f'Best checkpoint: reward={best_mean_reward:.3f}±{best_std_reward:.3f}, '
                f'len={best_mean_len:.2f}'
            )
            if best_mean_reward >= mean_reward:
                selected_model = best_model
                selected_label = 'best'
                mean_reward, std_reward, mean_len = best_mean_reward, best_std_reward, best_mean_len
        finally:
            best_eval_env.close()

    print(f'Using {selected_label} PPO checkpoint for saved artifact.')

    # Save model
    if args.save:
        save_path = os.path.join(save_dir, 'ppo_world_model')
        selected_model.save(save_path)
        print(f"Model saved to {save_path}")
        if legacy_save_dir is not None and os.path.abspath(legacy_save_dir) != os.path.abspath(save_dir):
            legacy_path = os.path.join(legacy_save_dir, 'ppo_world_model')
            selected_model.save(legacy_path)
            print(f"Legacy model alias saved to {legacy_path}")

    env.close()
    eval_env.close()
    return selected_model


if __name__ == '__main__':
    # Parse args
    args = get_args()

    # Setup wandb if enabled
    if args.wandb:
        import wandb

        wandb.init(
            project='discrete-model-only-rl',
            config=args,
            tags=args.tags,
            settings=wandb.Settings(start_method='thread'),
            allow_val_change=True
        )
        args = wandb.config

    # Train the model
    model = train_rl_model(args)
