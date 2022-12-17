import quaternion  # noqa
from gym.envs.registration import register

register(
    id="grasp-tabletop-v0",
    entry_point="thesis.envs:GraspTabletopEnv",
    max_episode_steps=200,
)

register(
    id="peg-insertion-v0",
    entry_point="thesis.envs:PegInsertionEnv",
    max_episode_steps=200,
)

register(
    id="play-table-v0", entry_point="thesis.envs:PlayTableEnv", max_episode_steps=200
)

register(
    id="goal-conditioned-v0",
    entry_point="thesis.envs:GoalConditionedEnv",
    max_episode_steps=200,
)
