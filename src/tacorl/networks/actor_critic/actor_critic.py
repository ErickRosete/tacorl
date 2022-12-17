import hydra
import torch.nn as nn
from gym import Env
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

from tacorl.utils.gym_utils import get_env_info, make_env


class ActorCritic(nn.Module):
    def __init__(
        self,
        env: Env,
        actor: DictConfig = {},
        critic: DictConfig = {},
    ):
        """ "
        Args
            encoder_path: If the used path is 'imagenet' it will load the resnet models
                          pretrained in imagenet, if a custom path is specified the
                          corresponding pretrained model will be loaded
        """

        super(ActorCritic, self).__init__()
        self.env_info = get_env_info(env)

        actor_cfg = OmegaConf.to_container(actor, resolve=True)
        actor_cfg.update(self.env_info)
        self.actor = hydra.utils.instantiate(actor_cfg)

        critic_cfg = OmegaConf.to_container(critic, resolve=True)
        critic_cfg.update(self.env_info)
        self.critic = hydra.utils.instantiate(critic_cfg)
        self.critic_target = hydra.utils.instantiate(critic_cfg)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def forward(
        self,
        observation: dict,
        deterministic: bool = False,
        reparameterize: bool = False,
    ):

        action, log_pi, z = self.actor.get_actions(
            observation=observation,
            deterministic=deterministic,
            reparameterize=reparameterize,
        )
        action_value = self.critic(observation=observation, action=action).detach()
        return action_value, action, log_pi, z


@hydra.main(config_path="../../../config", config_name="test/networks_test")
def main(cfg):
    actor_critic_cfg = OmegaConf.to_container(cfg.actor_critic, resolve=True)
    actor_critic_cfg["env"] = make_env(cfg.env)
    actor_critic = hydra.utils.instantiate(actor_critic_cfg)
    print(actor_critic)


if __name__ == "__main__":
    main()
