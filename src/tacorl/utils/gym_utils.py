import gym
from gym import Env
from stable_baselines3.common.utils import set_random_seed


def make_env(env_cfg):
    sel_env_spec = None
    all_envs = gym.envs.registry.all()
    for env_spec in all_envs:
        if env_cfg.name in env_spec.id:
            sel_env_spec = env_spec
            break

    if sel_env_spec is None:
        return None

    if "thesis" in sel_env_spec.entry_point:
        env = gym.make(sel_env_spec.id, **env_cfg)
        env._max_episode_steps = env_cfg.max_episode_steps
    else:
        env = gym.make(sel_env_spec.id)
    return env


def make_env_fn(cfg, rank, seed=0):
    def _init():
        env = make_env(cfg)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def get_env_info(env: Env):
    env_modalities = env.modalities if hasattr(env, "modalities") else []
    goal_modalities = env.goal_modalities if hasattr(env, "goal_modalities") else []

    proprioceptive_dim, scene_dim = 0, 0
    if len(goal_modalities) == 0 and len(env_modalities) == 0:
        # Gym env
        proprioceptive_dim = env.observation_space.shape[0]
    else:
        # Custom Gym env (Calvin)
        if "robot_obs" in env_modalities or "robot_obs" in goal_modalities:
            proprioceptive_dim = env.proprioceptive_dim
        if "scene_obs" in env_modalities or "scene_obs" in goal_modalities:
            scene_dim = env.scene_dim

    env_info = {
        "proprioceptive_dim": proprioceptive_dim,
        "scene_dim": scene_dim,
        "env_modalities": env_modalities,
        "goal_modalities": goal_modalities,
        "action_dim": env.action_space.shape[0],
    }
    return env_info
