import logging
import math
from typing import Optional, Tuple

import torch
import torch.distributions as D
import torch.nn as nn

import tacorl.networks.action_decoders.rnn_models as rnn_models
from tacorl.networks.action_decoders.action_decoder import ActionDecoder

logger = logging.getLogger(__name__)

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5


class ActionDecoderGaussian(ActionDecoder):
    def __init__(
        self,
        state_dim: int = 32,
        goal_dim: int = 32,
        latent_plan_dim: int = 16,
        hidden_size: int = 256,
        out_features: int = 7,
        policy_rnn_dropout_p: float = 0.0,
        num_layers: int = 2,
        rnn_model: str = "lstm_decoder",
        n_mixtures: int = 10,
        include_goal: bool = False,
    ):
        super(ActionDecoderGaussian, self).__init__()
        self.latent_plan_dim = latent_plan_dim
        self.include_goal = include_goal

        in_features = state_dim + latent_plan_dim
        if self.include_goal:
            in_features += goal_dim

        self.rnn = getattr(rnn_models, rnn_model)
        self.rnn = self.rnn(in_features, hidden_size, num_layers, policy_rnn_dropout_p)
        self.gaussian_mixture_model = MDN(
            in_features=hidden_size,
            out_features=out_features,
            n_gaussians=n_mixtures,
            log_scale_min=LOG_SIG_MIN,
            log_scale_max=LOG_SIG_MAX,
        )
        self.hidden_state = None

    def clear_hidden_state(self) -> None:
        self.hidden_state = None

    def loss_and_act(  # type:  ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        actions: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi, sigma, mu, _ = self(latent_plan, perceptual_emb, latent_goal)
        # loss
        loss = self.gaussian_mixture_model.loss(pi, sigma, mu, actions)
        # act
        pred_actions = self._sample(pi, sigma, mu)
        return loss, pred_actions

    def act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pi, sigma, mu, self.hidden_state = self(
            latent_plan, perceptual_emb, latent_goal, self.hidden_state
        )
        pred_actions = self._sample(pi, sigma, mu)
        return pred_actions

    def loss(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        actions: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pi, sigma, mu, _ = self(latent_plan, perceptual_emb, latent_goal)
        return self.gaussian_mixture_model.loss(pi, sigma, mu, actions)

    def _sample(self, *args, **kwargs):
        return self.gaussian_mixture_model.sample(*args, **kwargs)

    def forward(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # perceptual_emb = perceptual_emb[..., slice(*self.perceptual_emb_slice)]
        seq_len = perceptual_emb.shape[1]
        latent_plan = latent_plan.unsqueeze(1).expand(-1, seq_len, -1)
        latent_goal = latent_goal.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat(
            [latent_plan, perceptual_emb, latent_goal], dim=-1
        )  # b, s, (plan + visuo-propio + goal)
        if not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase):
            x, h_n = self.rnn(x, h_0)
        else:
            x = self.rnn(x)
            h_n = None
        pi, std, mu = self.gaussian_mixture_model(x)
        return pi, std, mu, h_n


class MDN(nn.Module):
    """Mixture Density Network - Gaussian Mixture Model, see Bishop, 1994
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        n_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxSxD): B is the batch size, S sequence length and D
        is the number of input dimensions.
    Output:
        (pi, sigma, mu) (BxSxK, BxSxKxO, BxSxKxO): B is the batch size,
            S sequence length, K is the number of Gaussians,
            and O is the number of dimensions for each Gaussian.
            Pi is a multinomial distribution of the Gaussians.
            Sigma is the standard deviation of each Gaussian.
            Mu is the mean of each  Gaussian.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_gaussians: int,
        log_scale_min: float = -7.0,
        log_scale_max: float = 7.0,
    ):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_gaussians = n_gaussians
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.pi = nn.Sequential(  # priors - Softmax guarantees sum = 1
            nn.Linear(in_features, n_gaussians), nn.Softmax(dim=-1)
        )
        self.log_var = nn.Linear(in_features, out_features * n_gaussians)
        self.mu = nn.Linear(in_features, out_features * n_gaussians)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape[0], x.shape[1]
        pi = self.pi(x)  # b, s, k number of gaussians
        log_var = self.log_var(x)
        log_scales = torch.clamp(
            log_var, min=self.log_scale_min, max=self.log_scale_max
        )  # avoid going to -inf / +inf
        std = torch.exp(log_scales)  # Guarantees that sigma is positive
        std = std.view(
            batch_size, seq_len, self.n_gaussians, self.out_features
        )  # b, s, k, o
        mu = self.mu(x)
        mu = mu.view(
            batch_size, seq_len, self.n_gaussians, self.out_features
        )  # b, s, k, o
        return pi, std, mu

    def loss(
        self,
        pi: torch.Tensor,
        sigma: torch.Tensor,
        mu: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the error, given the MoG parameters and the target
        The loss is the negative log likelihood of the data given the MoG
        parameters.
        """
        gmm = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=pi),
            component_distribution=D.Independent(D.Normal(mu, sigma), 1),
        )
        log_probs = gmm.log_prob(target)
        return -torch.mean(log_probs)

    def sample(
        self, pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        gmm = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=pi),
            component_distribution=D.Independent(D.Normal(mu, sigma), 1),
        )
        return gmm.sample()
