import cv2
import gym
import numpy as np
import pytorch_lightning as pl
import torch

from tacorl.modules.cem.cem import CEMOptimizer
from tacorl.utils.misc import add_img_thumbnail_to_imgs, extract_img_from_obs
from tacorl.utils.transforms import TransformManager, transform_observation
from tacorl.utils.wandb_loggers.video_logger import VideoLogger


class BaseRolloutManager(object):
    """Giving a pl_module and an Env, it can perform an episode rollout"""

    def __init__(self, transform_manager: TransformManager) -> None:
        self.transform_manager = transform_manager

    @staticmethod
    def get_critic(pl_module: pl.LightningModule):
        if hasattr(pl_module, "critic"):
            critic = pl_module.critic
        elif hasattr(pl_module, "q1"):
            q1, q2 = pl_module.q1, pl_module.q2
            return q1, q2
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
        reset_info: dict = {},
        render: bool = False,
        video_logger: VideoLogger = None,
        log_video: bool = False,
        task: str = None,
    ):
        pass

    def save_video(self, video_filename: str, images: np.ndarray, fps: int = 15):
        """
        Saves rollout video
        images: np.array, images used to create the video
                shape - seq, channels, height, width
        video_filename: str, path used to saved the video file
        """
        output_video = cv2.VideoWriter(
            video_filename,
            cv2.VideoWriter_fourcc(*"MP4V"),
            fps,
            images.shape[-2:],
        )

        images = np.moveaxis(images, 1, -1)[..., ::-1]
        for img in images:
            output_video.write(img)

        output_video.release()


class RLRollout(BaseRolloutManager):
    def __init__(self, use_cem: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_cem = use_cem

    def episode_rollout(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        reset_info: dict = {},
        render: bool = False,
        video_logger: VideoLogger = None,
        log_video: bool = False,
        task: str = None,
        save_video: bool = False,
        video_filename: str = "rollout.mp4",
    ):
        actor = self.get_actor(pl_module)
        if self.use_cem:
            q1, q2 = self.get_critic(pl_module)
            cem = CEMOptimizer(
                q1=q1,
                q2=q2,
                action_dim=actor.action_dim,
                discrete_gripper=actor.discrete_gripper,
            )
        # Tracking vars
        episode_return = 0

        # Episode rollout
        observation = env.reset(**reset_info)

        # Saving video or logging video
        if save_video or log_video:
            initial_img = extract_img_from_obs(observation)
            if save_video:
                img_list = [initial_img]
            if log_video:
                video_logger.new_video(initial_img, task=task)

        for step in range(1, env._max_episode_steps + 1):
            transf_obs = transform_observation(
                transform_manager=self.transform_manager,
                obs=observation,
                transf_type="validation",
                device=pl_module.device,
            )
            if self.use_cem:
                initial_mean, _ = actor.get_actions(
                    transf_obs,
                    deterministic=True,
                    reparameterize=False,
                )
                action = (
                    cem.get_action(obs=transf_obs, initial_mean=initial_mean)
                    .cpu()
                    .numpy()
                )
            else:
                action = self.get_actions(actor=actor, transf_obs=transf_obs)
            observation, reward, done, info = env.step(action)
            episode_return += reward

            # Logging
            if render:
                env.render()
            if save_video or log_video:
                current_img = extract_img_from_obs(observation)
                if save_video:
                    img_list.append(current_img)
                if log_video:
                    video_logger.update(current_img)

            if done:
                break

        # Logging videos
        if save_video or log_video:
            if save_video:
                img_array = np.moveaxis(np.stack(img_list), -1, 1)
            if isinstance(observation, dict) and "goal" in observation:
                goal_img = extract_img_from_obs(observation["goal"])
                if save_video:
                    img_array = add_img_thumbnail_to_imgs(img_array, goal_img)
                if log_video:
                    video_logger.add_goal_thumbnail(goal_img)
            if log_video:
                video_logger.write_to_tmp()
            if save_video:
                self.save_video(video_filename, img_array)

        rollout_info = {
            "episode_length": step,
            "episode_return": episode_return,
            "success": ("success" in info) and info["success"],
        }
        if "successful_tasks" in info:
            rollout_info["successful_tasks"] = info["successful_tasks"]

        return rollout_info


class LatentPlanRollout(BaseRolloutManager):
    def __init__(self, plan_duration: int = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.plan_duration = plan_duration

    def episode_rollout(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        reset_info: dict = {},
        render: bool = False,
        video_logger: VideoLogger = None,
        log_video: bool = False,
        task: str = None,
        save_video: bool = False,
        video_filename: str = "rollout.mp4",
    ):
        # Networks
        plan_proposal = pl_module.plan_proposal
        goal_encoder = pl_module.goal_encoder
        perceptual_encoder = pl_module.perceptual_encoder
        action_decoder = pl_module.action_decoder
        # Modalities in networks
        state_modalities = list(
            set(
                pl_module.plan_proposal_obs_modalities
                + pl_module.action_decoder_modalities
            )
        )
        # Tracking vars
        episode_return = 0

        # Episode rollout
        observation = env.reset(**reset_info)
        transf_obs = transform_observation(
            transform_manager=self.transform_manager,
            obs=observation,
            transf_type="validation",
            device=pl_module.device,
        )

        # Saving video or logging video
        if save_video or log_video:
            initial_img = extract_img_from_obs(observation)
            if save_video:
                img_list = [initial_img]
            if log_video:
                video_logger.new_video(initial_img, task=task)

        step, done = 0, False
        while not done and step < env._max_episode_steps:
            emb_state = perceptual_encoder.get_state_from_observation(
                observation=transf_obs["observation"],
                modalities=state_modalities,
                cat_output=False,
            )
            pp_state = torch.cat(
                [emb_state[key] for key in pl_module.plan_proposal_obs_modalities],
                dim=-1,
            )
            pp_goal = perceptual_encoder.get_state_from_observation(
                observation=transf_obs["goal"],
                modalities=pl_module.plan_proposal_goal_modalities,
            )
            pp_goal = goal_encoder(pp_goal)
            pp_dist = plan_proposal.get_dist(state_emb=pp_state, goal_emb=pp_goal)
            latent_plan = pp_dist.sample()
            action_decoder.clear_hidden_state()

            for _ in range(self.plan_duration):
                ad_state = perceptual_encoder.get_state_from_observation(
                    observation=transf_obs["observation"],
                    modalities=pl_module.action_decoder_modalities,
                )
                action = action_decoder.act(
                    latent_plan=latent_plan.unsqueeze(0),
                    perceptual_emb=ad_state.unsqueeze(0).unsqueeze(0),
                )
                action = action.squeeze().cpu().numpy()
                observation, reward, done, info = env.step(action)
                transf_obs = transform_observation(
                    transform_manager=self.transform_manager,
                    obs=observation,
                    transf_type="validation",
                    device=pl_module.device,
                )
                episode_return += reward
                step += 1
                # Logging
                if render:
                    env.render()
                if save_video or log_video:
                    current_img = extract_img_from_obs(observation)
                    if save_video:
                        img_list.append(current_img)
                    if log_video:
                        video_logger.update(current_img)

                if done or step >= env._max_episode_steps:
                    break

        # Logging videos
        if save_video or log_video:
            if save_video:
                img_array = np.moveaxis(np.stack(img_list), -1, 1)
            if isinstance(observation, dict) and "goal" in observation:
                goal_img = extract_img_from_obs(observation["goal"])
                if save_video:
                    img_array = add_img_thumbnail_to_imgs(img_array, goal_img)
                if log_video:
                    video_logger.add_goal_thumbnail(goal_img)
            if log_video:
                video_logger.write_to_tmp()
            if save_video:
                self.save_video(video_filename, img_array)

        rollout_info = {
            "episode_length": step,
            "episode_return": episode_return,
            "success": ("success" in info) and info["success"],
        }
        if "successful_tasks" in info:
            rollout_info["successful_tasks"] = info["successful_tasks"]

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
        reset_info: dict = {},
        render: bool = False,
        video_logger: VideoLogger = None,
        log_video: bool = False,
        task: str = None,
        save_video: bool = False,
        video_filename: str = "rollout.mp4",
    ):
        pp_actor = self.get_actor(pl_module)
        if self.use_cem:
            q1, q2 = self.get_critic(pl_module)
            cem = CEMOptimizer(
                q1=q1,
                q2=q2,
                action_dim=pp_actor.action_dim,
            )
        perceptual_encoder = pl_module.perceptual_encoder
        action_decoder = pl_module.action_decoder

        # Tracking vars
        episode_return = 0

        # Episode rollout
        observation = env.reset(**reset_info)
        transf_obs = transform_observation(
            transform_manager=self.transform_manager,
            obs=observation,
            transf_type="validation",
            device=pl_module.device,
        )

        # Saving video or logging video
        if save_video or log_video:
            initial_img = extract_img_from_obs(observation)
            if save_video:
                img_list = [initial_img]
            if log_video:
                video_logger.new_video(initial_img, task=task)

        step, done = 0, False
        while not done and step < env._max_episode_steps:
            if self.use_cem:
                initial_mean, _ = pp_actor.get_actions(
                    transf_obs,
                    deterministic=True,
                    reparameterize=False,
                )
                latent_plan = cem.get_action(obs=transf_obs, initial_mean=initial_mean)
            else:
                latent_plan, _ = pp_actor.get_actions(
                    transf_obs,
                    deterministic=True,
                    reparameterize=False,
                )
            # For action decoder
            action_decoder.clear_hidden_state()
            for _ in range(self.plan_duration):
                ad_state = perceptual_encoder.get_state_from_observation(
                    observation=transf_obs["observation"],
                    modalities=pl_module.action_decoder_modalities,
                )
                action = action_decoder.act(
                    latent_plan=latent_plan.unsqueeze(0),
                    perceptual_emb=ad_state.unsqueeze(0).unsqueeze(0),
                )
                action = action.squeeze().cpu().numpy()
                observation, reward, done, info = env.step(action)
                transf_obs = transform_observation(
                    transform_manager=self.transform_manager,
                    obs=observation,
                    transf_type="validation",
                    device=pl_module.device,
                )
                episode_return += reward
                step += 1
                if render:
                    env.render()
                if save_video or log_video:
                    current_img = extract_img_from_obs(observation)
                    if save_video:
                        img_list.append(current_img)
                    if log_video:
                        video_logger.update(current_img)

                if done or step >= env._max_episode_steps:
                    break

        if save_video or log_video:
            if save_video:
                img_array = np.moveaxis(np.stack(img_list), -1, 1)
            if isinstance(observation, dict) and "goal" in observation:
                goal_img = extract_img_from_obs(observation["goal"])
                if save_video:
                    img_array = add_img_thumbnail_to_imgs(img_array, goal_img)
                if log_video:
                    video_logger.add_goal_thumbnail(goal_img)
            if log_video:
                video_logger.write_to_tmp()
            if save_video:
                self.save_video(video_filename, img_array)

        rollout_info = {
            "episode_length": step,
            "episode_return": episode_return,
            "success": ("success" in info) and info["success"],
        }
        if "successful_tasks" in info:
            rollout_info["successful_tasks"] = info["successful_tasks"]

        return rollout_info


class RelayImitationLearning(BaseRolloutManager):
    def __init__(self, plan_duration: int = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.plan_duration = plan_duration

    def episode_rollout(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        reset_info: dict = {},
        render: bool = False,
        video_logger: VideoLogger = None,
        log_video: bool = False,
        task: str = None,
        save_video: bool = False,
        video_filename: str = "rollout.mp4",
    ):
        # Networks
        goal_encoder = pl_module.goal_encoder
        perceptual_encoder = pl_module.perceptual_encoder
        high_level_policy = pl_module.high_level_policy
        low_level_policy = pl_module.low_level_policy

        # Modalities in networks
        all_modalities = pl_module.all_modalities
        # Tracking vars
        episode_return = 0

        # Episode rollout
        observation = env.reset(**reset_info)
        transf_obs = transform_observation(
            transform_manager=self.transform_manager,
            obs=observation,
            transf_type="validation",
            device=pl_module.device,
        )

        # Saving video or logging video
        if save_video or log_video:
            initial_img = extract_img_from_obs(observation)
            if save_video:
                img_list = [initial_img]
            if log_video:
                video_logger.new_video(initial_img, task=task)

        step, done = 0, False
        while not done and step < env._max_episode_steps:
            emb_state = perceptual_encoder.get_state_from_observation(
                observation=transf_obs["observation"],
                modalities=all_modalities,
                cat_output=False,
            )
            emb_state = torch.cat(
                [emb_state[key] for key in pl_module.high_level_policy_modalities],
                dim=-1,
            )
            emb_goal = perceptual_encoder.get_state_from_observation(
                observation=transf_obs["goal"],
                modalities=pl_module.high_level_policy_modalities,
            )
            emb_goal = goal_encoder(emb_goal)

            high_level_obs = torch.cat([emb_state, emb_goal], dim=-1)
            latent_subgoal, _ = high_level_policy.get_actions(
                high_level_obs, deterministic=True
            )

            for _ in range(self.plan_duration):
                emb_state = perceptual_encoder.get_state_from_observation(
                    observation=transf_obs["observation"],
                    modalities=pl_module.low_level_policy_modalities,
                )

                low_level_obs = torch.cat([emb_state, latent_subgoal], dim=-1)
                action, _ = low_level_policy.get_actions(
                    low_level_obs, deterministic=True
                )
                action = action.squeeze().cpu().numpy()
                observation, reward, done, info = env.step(action)
                transf_obs = transform_observation(
                    transform_manager=self.transform_manager,
                    obs=observation,
                    transf_type="validation",
                    device=pl_module.device,
                )
                episode_return += reward
                step += 1
                # Logging
                if render:
                    env.render()
                if save_video or log_video:
                    current_img = extract_img_from_obs(observation)
                    if save_video:
                        img_list.append(current_img)
                    if log_video:
                        video_logger.update(current_img)

                if done or step >= env._max_episode_steps:
                    break

        # Logging videos
        if save_video or log_video:
            if save_video:
                img_array = np.moveaxis(np.stack(img_list), -1, 1)
            if isinstance(observation, dict) and "goal" in observation:
                goal_img = extract_img_from_obs(observation["goal"])
                if save_video:
                    img_array = add_img_thumbnail_to_imgs(img_array, goal_img)
                if log_video:
                    video_logger.add_goal_thumbnail(goal_img)
            if log_video:
                video_logger.write_to_tmp()
            if save_video:
                self.save_video(video_filename, img_array)

        rollout_info = {
            "episode_length": step,
            "episode_return": episode_return,
            "success": ("success" in info) and info["success"],
        }
        if "successful_tasks" in info:
            rollout_info["successful_tasks"] = info["successful_tasks"]

        return rollout_info
