import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
from threading import Lock
from collections import deque

import numpy as np
from gymnasium import spaces
import gymnasium
import gym
# Force gym to use gymnasium spaces
gym.spaces = gymnasium.spaces

import h5py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env_helpers import *
from model_construction import construct_ae_model
from shared.models import SB3GeneralEncoder, SB3ActorCriticPolicy
from shared.models import mlp
from training_helpers import vec_env_random_walk, vec_env_ez_explore
from tqdm import tqdm
from model_free.rl_utils import interpret_layer_sizes


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env_name', type=str, default='MiniGrid-MultiRoom-N2-S4-v0')
parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
parser.add_argument('-n', '--n_envs', type=int, default=8)
parser.add_argument('-s', '--train_steps', type=int, default=int(3e5))
parser.add_argument('-c', '--chunk_size', type=int, default=2048)
parser.add_argument('-ct', '--compression_type', type=str, default='lzf')
# ppo, ppo_entropy, ppo_rollout, model_free_rollout, random, ezexplore, bfs
parser.add_argument('-a', '--algorithm', type=str, default='random')
parser.add_argument('--extra_info', nargs='*', default=[])
parser.add_argument('--norm_stats', action='store_true')
parser.add_argument('--shrink_size', type=int, default=None,
    help='If given, the final dataset will be shrunken down to this size by '
         'taking half from the start of training and the other half from the end.')
parser.add_argument('--env_max_steps', type=int, default=None,
                    help='Maximum steps per episode in the environment')
parser.add_argument('--random_seeds', action='store_true',
                    help='Use diverse random seeds for each episode (better generalization)')
parser.set_defaults(random_seeds=False)
parser.set_defaults(norm_stats=False)

# New: SB3 model save/load for Option A
parser.add_argument('--save_sb3_path', type=str, default=None,
                    help='Where to save an SB3 PPO model (e.g., ./sb3_agent.zip)')
parser.add_argument('--load_sb3_path', type=str, default=None,
                    help='Where to load an SB3 PPO model (e.g., ./sb3_agent.zip)')
parser.add_argument('--load_model_free_path', type=str, default=None,
                    help='Path to a saved custom model_free checkpoint (.pt) for rollout collection')
parser.add_argument('--deterministic', action='store_true',
                    help='Use deterministic actions when rolling out a loaded policy')
parser.add_argument('--ae_model_type', type=str, default='vqvae')
parser.add_argument('--ae_model_version', type=str, default='2')
parser.add_argument('--ae_model_hash', type=str, default=None)
parser.add_argument('--codebook_size', type=int, default=64)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--filter_size', type=int, default=9)
parser.add_argument('--latent_dim', type=int, default=None)
parser.add_argument('--device', type=str, default='cuda')


def _split_step(step_result):
    """
    Handle both gym (obs, reward, done, info) and gymnasium
    (obs, reward, terminated, truncated, info).
    """
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return obs, reward, done, info
    else:
        obs, reward, done, info = step_result
        return obs, reward, bool(done), info


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, replay_buffer, buffer_lock):
        super(EarlyStoppingCallback, self).__init__(verbose=0)
        self.replay_buffer = replay_buffer
        self.lock = buffer_lock
        self.check_interval = 10000
        self.idx = 0

    def _on_step(self) -> bool:
        self.idx += 1
        if self.idx % self.check_interval != 0:
            return True

        with self.lock:
            n_samples = int(self.replay_buffer.attrs['data_idx'])
            buffer_size = int(self.replay_buffer['obs'].shape[0])
            if n_samples >= buffer_size:
                print('Enough data collected, stopping training')
                return False

        print(f'{n_samples}/{buffer_size} transitions recorded')
        return True


def buffer_full(replay_buffer, lock: Lock) -> bool:
    with lock:
        return int(replay_buffer.attrs['data_idx']) >= int(replay_buffer['obs'].shape[0])


def setup_replay_buffer(args, path=None):
    sanitized_env_name = args.env_name.replace(':', '_')
    replay_buffer_path = path or f'./data/{sanitized_env_name}_replay_buffer.hdf5'
    os.makedirs(os.path.dirname(replay_buffer_path), exist_ok=True)

    replay_buffer = h5py.File(replay_buffer_path, 'w')

    # IMPORTANT: keep semantics as original: train_steps == total transitions desired
    dataset_size = int(args.train_steps)

    env = make_env(args.env_name, max_steps=args.env_max_steps)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    act_type = 'int32' if isinstance(env.action_space, spaces.Discrete) else 'float32'

    replay_buffer.create_dataset(
        'obs', (dataset_size, *obs_shape),
        maxshape=(dataset_size, *obs_shape),
        compression=args.compression_type, dtype='float16',
        chunks=(min(args.chunk_size, dataset_size), *obs_shape)
    )

    if len(act_shape) == 0:
        replay_buffer.create_dataset(
            'action', (dataset_size,),
            maxshape=(dataset_size,),
            compression=args.compression_type, dtype=act_type,
            chunks=(min(args.chunk_size, dataset_size),)
        )
    else:
        replay_buffer.create_dataset(
            'action', (dataset_size, *act_shape),
            maxshape=(dataset_size, *act_shape),
            compression=args.compression_type, dtype=act_type,
            chunks=(min(args.chunk_size, dataset_size), *act_shape)
        )

    replay_buffer.create_dataset(
        'next_obs', (dataset_size, *obs_shape),
        maxshape=(dataset_size, *obs_shape),
        compression=args.compression_type, dtype='float16',
        chunks=(min(args.chunk_size, dataset_size), *obs_shape)
    )

    replay_buffer.create_dataset(
        'reward', (dataset_size,),
        maxshape=(dataset_size,),
        compression=args.compression_type, dtype='float32',
        chunks=(min(args.chunk_size, dataset_size),)
    )

    replay_buffer.create_dataset(
        'done', (dataset_size,),
        maxshape=(dataset_size,),
        compression=args.compression_type, dtype='bool',
        chunks=(min(args.chunk_size, dataset_size),)
    )

    replay_buffer.attrs['data_idx'] = 0

    # Extra info keys probing (robust to gymnasium)
    if args.extra_info:
        _ = env.reset()
        step_result = env.step(env.action_space.sample())
        _, _, _, info = _split_step(step_result)

        for key in args.extra_info:
            assert key in info, f'Key {key} not in info dict!'
            arr = np.array(info[key])
            dtype = 'int16' if 'int' in str(arr.dtype) else 'float16'
            print(f'Found key `{key}` with shape {arr.shape} and dtype {dtype}')
            replay_buffer.create_dataset(
                key, (dataset_size, *arr.shape),
                maxshape=(dataset_size, *arr.shape),
                compression=args.compression_type, dtype=dtype,
                chunks=(min(args.chunk_size, dataset_size), *arr.shape)
            )

    env.close()
    return replay_buffer


def shrink_replay_buffer(replay_buffer, new_size):
    half_size = int(new_size / 2)
    for key in replay_buffer.keys():
        # Make the second half of the buffer the newest data
        replay_buffer[key][new_size-half_size:new_size] = replay_buffer[key][-half_size:]
        if len(replay_buffer[key].shape) == 1:
            replay_buffer[key].resize((new_size,))
        else:
            replay_buffer[key].resize((new_size, *replay_buffer[key].shape[1:]))
    replay_buffer.attrs['data_idx'] = min(int(replay_buffer.attrs['data_idx']), int(new_size))


# =========================
# Option B: BFS Solver
# =========================

def _find_goal(env_u):
    w, h = int(env_u.width), int(env_u.height)
    for x in range(w):
        for y in range(h):
            obj = env_u.grid.get(x, y)
            if obj is not None and getattr(obj, "type", None) == "goal":
                return (x, y)
    return None


def _is_blocked(env_u, x, y):
    obj = env_u.grid.get(x, y)
    if obj is None:
        return False
    t = getattr(obj, "type", "")
    return t in ("wall", "lava")


def _bfs_path(env_u, start, goal):
    q = deque([start])
    prev = {start: None}
    w, h = int(env_u.width), int(env_u.height)

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            break
        for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if _is_blocked(env_u, nx, ny):
                continue
            if (nx, ny) not in prev:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))

    if goal not in prev:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path  # includes start


def _turn_action(env_u, target_dir):
    acts = getattr(env_u, "actions", None)
    left = int(getattr(acts, "left", 0)) if acts is not None else 0
    right = int(getattr(acts, "right", 1)) if acts is not None else 1

    cur = int(env_u.agent_dir)
    diff = (target_dir - cur) % 4
    if diff == 0:
        return None
    if diff == 1:
        return right
    if diff == 3:
        return left
    # diff == 2: turn twice, do right once now
    return right


def bfs_policy_action(env_u):
    """
    Returns a single discrete action that greedily follows a BFS shortest path
    to the goal on the TRUE grid. This is an "expert" demonstrator.
    """
    start = tuple(env_u.agent_pos)
    goal = _find_goal(env_u)
    if goal is None:
        return env_u.action_space.sample()

    path = _bfs_path(env_u, start, goal)
    if path is None or len(path) < 2:
        return env_u.action_space.sample()

    nx, ny = path[1]
    cx, cy = start
    dx, dy = nx - cx, ny - cy

    # desired direction (MiniGrid convention: 0:right,1:down,2:left,3:up)
    if (dx, dy) == (1, 0):
        target_dir = 0
    elif (dx, dy) == (0, 1):
        target_dir = 1
    elif (dx, dy) == (-1, 0):
        target_dir = 2
    elif (dx, dy) == (0, -1):
        target_dir = 3
    else:
        return env_u.action_space.sample()

    turn = _turn_action(env_u, target_dir)
    if turn is not None:
        return turn

    acts = getattr(env_u, "actions", None)
    forward = int(getattr(acts, "forward", 2)) if acts is not None else 2
    return forward


def _split_reset(reset_result):
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _strip_compile_prefix(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_model_free_policy_from_checkpoint(model_path, encoder, args, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})

    if args.ae_model_type == 'vqvae':
        input_dim = args.embedding_dim * encoder.n_latent_embeds
    else:
        input_dim = encoder.latent_dim

    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_dim = env.action_space.n
    env.close()

    policy_hidden = interpret_layer_sizes(saved_args.get('policy_hidden', [256, 256]))
    rl_activation = saved_args.get('rl_activation', 'relu')
    vqvae_e2e = (saved_args.get('ae_model_type') == 'vqvae' and saved_args.get('e2e_loss', False))

    mlp_kwargs = {
        'activation': rl_activation,
        'discrete_input': (args.ae_model_type == 'vqvae') and (not vqvae_e2e),
    }
    if args.ae_model_type == 'vqvae' and not vqvae_e2e:
        mlp_kwargs['n_embeds'] = args.codebook_size
        mlp_kwargs['embed_dim'] = args.embedding_dim

    policy = mlp([input_dim] + policy_hidden + [act_dim], **mlp_kwargs)
    policy.load_state_dict(_strip_compile_prefix(checkpoint['policy_state_dict']))
    return policy.to(device).eval()


# =========================
# Main
# =========================

if __name__ == '__main__':
    args = parser.parse_args()
    args.env_name = check_env_name(args.env_name)

    replay_buffer = setup_replay_buffer(args)
    buffer_lock = Lock()

    # Recording happens inside make_env wrapper when replay_buffer is provided
    venv = DummyVecEnv([lambda: make_env(
        args.env_name, replay_buffer, buffer_lock,
        extra_info=args.extra_info, monitor=True,
        max_steps=args.env_max_steps,
        random_seeds=args.random_seeds)] * args.n_envs)

    algo = args.algorithm.lower()

    # ---- Option A (training): PPO / PPO_ENTROPY
    if algo.startswith('ppo') and algo not in ('ppo_rollout',):
        policy_kwargs = dict(
            # Encoder
            features_extractor_class=SB3GeneralEncoder,
            features_extractor_kwargs={'features_dim': 256, 'fta': False},
            # Policy / Value nets
            policy_fta=False,
            critic_fta=False,
            hidden_sizes=[256],
        )
        entropy_coef = 0.01 if algo.endswith('entropy') else 0.0

        model = PPO(
            SB3ActorCriticPolicy,
            venv,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=None,
            learning_rate=args.learning_rate,
            ent_coef=entropy_coef,
        )

        # Train until buffer is full (via callback)
        model.learn(
            total_timesteps=int(args.train_steps) + 2048 * int(args.n_envs),
            log_interval=1,
            callback=EarlyStoppingCallback(replay_buffer, buffer_lock)
        )

        if args.save_sb3_path:
            model.save(args.save_sb3_path)
            print(f"Saved SB3 model to {args.save_sb3_path}")

    # ---- Option A (rollout-only): PPO_ROLLOUT (load SB3 model, deterministic actions)
    elif algo == 'ppo_rollout':
        assert args.load_sb3_path is not None, "Need --load_sb3_path for ppo_rollout"
        model = PPO.load(args.load_sb3_path, env=venv)
        obs = venv.reset()

        # Step until buffer full
        pbar = tqdm(total=int(replay_buffer['obs'].shape[0]), desc="Collecting (ppo_rollout)")
        last_n = 0

        while not buffer_full(replay_buffer, buffer_lock):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            if np.any(dones):
                obs = venv.reset()

            # update progress by samples recorded
            with buffer_lock:
                n = int(replay_buffer.attrs['data_idx'])
            if n > last_n:
                pbar.update(n - last_n)
                last_n = n

        pbar.close()
        print("Buffer full; rollout collection done.")

    # ---- Option A (rollout-only): custom model_free checkpoint
    elif algo == 'model_free_rollout':
        assert args.load_model_free_path is not None, "Need --load_model_free_path for model_free_rollout"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(args.load_model_free_path, map_location='cpu', weights_only=False)
        ckpt_args = ckpt.get('args', {})
        for key, value in ckpt_args.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        args.ae_model_type = ckpt_args.get('ae_model_type', args.ae_model_type)
        args.ae_model_version = ckpt_args.get('ae_model_version', args.ae_model_version)
        args.codebook_size = ckpt_args.get('codebook_size', args.codebook_size)
        args.embedding_dim = ckpt_args.get('embedding_dim', args.embedding_dim)
        args.filter_size = ckpt_args.get('filter_size', args.filter_size)
        args.latent_dim = ckpt_args.get('latent_dim', args.latent_dim)
        args.wandb = False
        args.comet_ml = False
        args.model_dir = '.'
        args.ae_grad_clip = getattr(args, 'ae_grad_clip', 0.0)
        sample_env = make_env(args.env_name, max_steps=args.env_max_steps)
        sample_obs = preprocess_obs([_split_reset(sample_env.reset())])
        encoder = construct_ae_model(sample_obs.shape[1:], args, load=False)[0]
        encoder.load_state_dict(ckpt['ae_model_state_dict'])
        encoder = encoder.to(device)
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        policy = load_model_free_policy_from_checkpoint(args.load_model_free_path, encoder, args, device)
        sample_env.close()

        obs = venv.reset()
        pbar = tqdm(total=int(replay_buffer['obs'].shape[0]), desc="Collecting (model_free_rollout)")
        last_n = 0

        while not buffer_full(replay_buffer, buffer_lock):
            obs_tensor = preprocess_obs(obs).to(device)
            with torch.no_grad():
                states = encoder.encode(obs_tensor, return_one_hot=True)
                logits = policy(states)
                if args.deterministic:
                    action = logits.argmax(dim=-1).cpu().numpy()
                else:
                    action = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

            obs, rewards, dones, infos = venv.step(action)
            if np.any(dones):
                obs = venv.reset()

            with buffer_lock:
                n = int(replay_buffer.attrs['data_idx'])
            if n > last_n:
                pbar.update(n - last_n)
                last_n = n

        pbar.close()
        print("Buffer full; model_free rollout collection done.")

    # ---- Option B: BFS scripted solver (expert trajectories)
    elif algo == 'bfs':
        obs = venv.reset()

        pbar = tqdm(total=int(replay_buffer['obs'].shape[0]), desc="Collecting (bfs)")
        last_n = 0

        while not buffer_full(replay_buffer, buffer_lock):
            actions = []
            for i in range(args.n_envs):
                env_i = venv.envs[i].unwrapped
                a = bfs_policy_action(env_i)
                actions.append(a)

            obs, rewards, dones, infos = venv.step(np.array(actions, dtype=np.int64))
            if np.any(dones):
                obs = venv.reset()

            with buffer_lock:
                n = int(replay_buffer.attrs['data_idx'])
            if n > last_n:
                pbar.update(n - last_n)
                last_n = n

        pbar.close()
        print("Buffer full; BFS collection done.")

    # ---- Existing: random / ezexplore
    elif algo == 'random':
        vec_env_random_walk(venv, args.train_steps)
    elif algo == 'ezexplore':
        vec_env_ez_explore(venv, args.train_steps)
    else:
        raise ValueError(f'Unknown algorithm: {args.algorithm}')

    # Shrink buffer if requested
    if args.shrink_size and args.shrink_size < int(replay_buffer['obs'].shape[0]):
        print('-------')
        print("Before shrink:", replay_buffer['obs'].shape)
        last_obs = replay_buffer['obs'][-1].copy()
        shrink_replay_buffer(replay_buffer, args.shrink_size)
        print("After shrink:", replay_buffer['obs'].shape)
        new_last_obs = replay_buffer['obs'][-1].copy()
        print("Last obs preserved:", (last_obs == new_last_obs).all())

    # Normalization stats (use actual recorded count)
    if args.norm_stats:
        print('Calculating normalization stats...')
        with buffer_lock:
            obs_count = int(replay_buffer.attrs['data_idx'])

        obs_sum = np.zeros(replay_buffer['obs'].shape[1:], dtype=np.float64)
        obs_square_sum = np.zeros(replay_buffer['obs'].shape[1:], dtype=np.float64)

        for i in tqdm(range(0, obs_count, args.chunk_size)):
            obs_batch = replay_buffer['obs'][i:i+args.chunk_size]
            obs_sum += obs_batch.sum(axis=0)
            obs_square_sum += np.square(obs_batch).sum(axis=0)

        obs_mean = obs_sum / max(obs_count, 1)
        # unbiased std, guard obs_count==1
        if obs_count > 1:
            var_num = (obs_count * obs_square_sum - np.square(obs_sum))
            var_den = (obs_count * (obs_count - 1))
            # Numerical guard: variance can go slightly negative from fp rounding.
            obs_var = np.maximum(var_num / max(var_den, 1), 1e-12)
            obs_std = np.sqrt(obs_var)
        else:
            obs_std = np.ones_like(obs_mean)

        replay_buffer.attrs['obs_mean'] = obs_mean.astype(np.float32)
        replay_buffer.attrs['obs_std'] = obs_std.astype(np.float32)

    try:
        venv.close()
    except Exception as e:
        print(f"⚠️  venv.close() failed: {e}")

    # 2) now it is safe to shrink / compute stats (file still open)
    if args.shrink_size and args.shrink_size < int(replay_buffer['obs'].shape[0]):
        shrink_replay_buffer(replay_buffer, args.shrink_size)

    if args.norm_stats:
        # compute stats...
        pass

    # 3) finally close the HDF5
    replay_buffer.close()
