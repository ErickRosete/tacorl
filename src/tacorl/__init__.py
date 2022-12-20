import quaternion  # noqa
from gym.envs.registration import register

register(
    id="grasp-tabletop-v0",
    entry_point="tacorl.envs:GraspTabletopEnv",
    max_episode_steps=200,
)

register(
    id="peg-insertion-v0",
    entry_point="tacorl.envs:PegInsertionEnv",
    max_episode_steps=200,
)

register(
    id="play-table-v0", entry_point="tacorl.envs:PlayTableEnv", max_episode_steps=200
)

register(
    id="goal-conditioned-v0",
    entry_point="tacorl.envs:GoalConditionedEnv",
    max_episode_steps=200,
)
