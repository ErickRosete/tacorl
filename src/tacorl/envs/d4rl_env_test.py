import d4rl  # noqa
import gym

env = gym.make("antmaze-large-diverse-v0")
# dataset = env.get_dataset()
for i in range(5):
    a = env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
