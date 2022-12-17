import gym
import pytorch_lightning as pl
import torch

from tacorl.modules.cem.cem import CEMOptimizer


class BaseRolloutManager(object):
    """Giving a pl_module and an Env, it can perform an episode rollout"""

    @staticmethod
    def get_critic(pl_module: pl.LightningModule):
        if hasattr(pl_module, "critic"):
            critic = pl_module.critic
        else:
            raise ValueError("Critic not found in pl module.")
        return critic

    @staticmethod
    def get_actor(pl_module: pl.LightningModule):
        if hasattr(pl_module, "actor"):
            actor = pl_module.actor
        else:
            raise ValueError("Actor not found in pl module.")
        return actor

    @staticmethod
    def get_actions(actor, transf_obs):
        """Return deterministic actions from the transf_obs"""
        actions, _ = actor.get_actions(
            transf_obs,
            deterministic=True,
            reparameterize=False,
        )
        return actions.detach().cpu().numpy()

    def episode_rollout(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        render: bool = False,
    ):
        pass


class RLRollout(BaseRolloutManager):
    def __init__(self, use_cem: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_cem = use_cem

    def episode_rollout(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        render: bool = False,
    ):
        actor = self.get_actor(pl_module)
        if self.use_cem:
            critic = self.get_critic(pl_module)
            cem = CEMOptimizer(
                critic=critic,
                action_dim=actor.action_dim,
                discrete_gripper=actor.discrete_gripper,
            )
        # Tracking vars
        episode_return = 0

        # Episode rollout
        observation = env.reset()
        goal = env.target_goal
        obs_tensor = torch.tensor(
            observation, device=pl_module.device, dtype=torch.float
        )
        goal_tensor = torch.tensor(goal, device=pl_module.device, dtype=torch.float)
        for step in range(1, env._max_episode_steps + 1):
            concat_obs_goal = torch.cat([obs_tensor, goal_tensor], dim=-1)
            if self.use_cem:
                initial_mean, _ = actor.get_actions(
                    concat_obs_goal,
                    deterministic=True,
                    reparameterize=False,
                )
                action = (
                    cem.get_action(obs=concat_obs_goal, initial_mean=initial_mean)
                    .cpu()
                    .numpy()
                )
            else:
                action = self.get_actions(actor=actor, transf_obs=concat_obs_goal)
            observation, reward, done, info = env.step(action)
            episode_return += reward

            # Logging
            if render:
                env.render()
            if done:
                break
        score = env.get_normalized_score(episode_return)
        rollout_info = {
            "episode_length": step,
            "episode_return": episode_return,
            "score": score,
            "success": ("success" in info) and info["success"],
        }
        return rollout_info


class LatentPlanRollout(BaseRolloutManager):
    def __init__(self, plan_duration: int = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.plan_duration = plan_duration

    def episode_rollout(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        render: bool = False,
    ):
        # Networks
        plan_proposal = pl_module.plan_proposal
        action_decoder = pl_module.action_decoder
        # Tracking vars
        episode_return = 0

        # Episode rollout
        observation = env.reset()
        if hasattr(env, "target_goal"):
            goal = env.target_goal
        elif hasattr(env, "goal_locations"):
            goal = env.goal_locations[0]

        obs_tensor = torch.tensor(
            observation, device=pl_module.device, dtype=torch.float
        )
        goal_tensor = torch.tensor(goal, device=pl_module.device, dtype=torch.float)
        step, done = 0, False
        while not done and step < env._max_episode_steps:
            pp_dist = plan_proposal.get_dist(state_emb=obs_tensor, goal_emb=goal_tensor)
            latent_plan = pp_dist.sample()
            action_decoder.clear_hidden_state()
            for _ in range(self.plan_duration):
                action = action_decoder.act(
                    latent_plan=latent_plan.unsqueeze(0),
                    perceptual_emb=obs_tensor.unsqueeze(0).unsqueeze(0),
                )
                action = action.squeeze().cpu().numpy()
                observation, reward, done, info = env.step(action)
                obs_tensor = torch.tensor(
                    observation, device=pl_module.device, dtype=torch.float
                )
                episode_return += reward
                step += 1
                # Logging
                if render:
                    env.render()
                if done or step >= env._max_episode_steps:
                    break

        score = env.get_normalized_score(episode_return)
        rollout_info = {
            "episode_length": step,
            "episode_return": episode_return,
            "score": score,
            "success": ("success" in info) and info["success"],
        }
        return rollout_info


class TACORL(BaseRolloutManager):
    def __init__(
        self, plan_duration: int = 16, use_cem: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.plan_duration = plan_duration
        self.use_cem = use_cem

    def episode_rollout(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        render: bool = False,
    ):
        pp_actor = self.get_actor(pl_module)
        if self.use_cem:
            critic = self.get_critic(pl_module)
            cem = CEMOptimizer(
                critic=critic,
                action_dim=pp_actor.action_dim,
            )
        action_decoder = pl_module.action_decoder

        # Tracking vars
        episode_return = 0

        # Episode rollout
        observation = env.reset()
        if hasattr(env, "target_goal"):
            goal = env.target_goal
        elif hasattr(env, "goal_locations"):
            goal = env.goal_locations[0]
        obs_tensor = torch.tensor(
            observation, device=pl_module.device, dtype=torch.float
        )
        goal_tensor = torch.tensor(goal, device=pl_module.device, dtype=torch.float)
        step, done = 0, False
        while not done and step < env._max_episode_steps:
            concat_obs_goal = torch.cat([obs_tensor, goal_tensor], dim=-1)
            if self.use_cem:
                initial_mean, _ = pp_actor.get_actions(
                    concat_obs_goal,
                    deterministic=True,
                    reparameterize=False,
                )
                latent_plan = cem.get_action(
                    obs=concat_obs_goal, initial_mean=initial_mean
                )
            else:
                latent_plan, _ = pp_actor.get_actions(
                    concat_obs_goal,
                    deterministic=True,
                    reparameterize=False,
                )
            # For action decoder
            action_decoder.clear_hidden_state()
            for _ in range(self.plan_duration):
                action = action_decoder.act(
                    latent_plan=latent_plan.unsqueeze(0),
                    perceptual_emb=obs_tensor.unsqueeze(0).unsqueeze(0),
                )
                action = action.squeeze().cpu().numpy()
                observation, reward, done, info = env.step(action)
                obs_tensor = torch.tensor(
                    observation, device=pl_module.device, dtype=torch.float
                )
                episode_return += reward
                step += 1
                # Logging
                if render:
                    env.render()
                if done or step >= env._max_episode_steps:
                    break

        score = env.get_normalized_score(episode_return)
        rollout_info = {
            "episode_length": step,
            "episode_return": episode_return,
            "score": score,
            "success": ("success" in info) and info["success"],
        }
        return rollout_info
