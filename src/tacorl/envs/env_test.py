import time

import hydra
from tqdm import tqdm

from tacorl.utils.gym_utils import make_env


@hydra.main(config_path="../../config", config_name="test/env_test")
def test_env(cfg):
    env = make_env(cfg.env)
    for _ in range(1000000):
        obs = env.reset()
        for i in tqdm(range(env.max_episode_steps)):
            obs, reward, done, info = env.step(env.action_space.sample())
            env.render()
            time.sleep(0.01)
            if done:
                break


if __name__ == "__main__":
    test_env()
