import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

import tacorl
import tacorl.networks.action_decoders.rnn_models as rnn_models
from tacorl.networks.action_decoders.action_decoder import ActionDecoder
from tacorl.utils.misc import log_sum_exp

logger = logging.getLogger(__name__)

LOG_SIG_MIN = -5


class ActionDecoderLogistic(ActionDecoder):
    def __init__(
        self,
        state_dim: int = 32,
        goal_dim: int = 32,
        latent_plan_dim: int = 16,
        hidden_size: int = 256,
        out_features: int = 7,
        act_max_bound: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        act_min_bound: List[float] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        gripper_alpha: float = 1.0,
        policy_rnn_dropout_p: float = 0.0,
        num_layers: int = 2,
        rnn_model: str = "rnn_decoder",
        discrete_gripper: bool = True,
        include_goal: bool = False,
        num_classes: int = 10,
        n_mixtures: int = 10,
    ):
        super(ActionDecoderLogistic, self).__init__()
        self.n_dist = n_mixtures
        self.discrete_gripper = discrete_gripper
        self.num_classes = num_classes
        self.latent_plan_dim = latent_plan_dim
        self.include_goal = include_goal

        in_features = state_dim + latent_plan_dim
        if self.include_goal:
            in_features += goal_dim

        self.out_features = (
            out_features - 1 if discrete_gripper else out_features
        )  # for discrete gripper act
        self.gripper_alpha = gripper_alpha
        self.rnn = getattr(rnn_models, rnn_model)
        self.rnn = self.rnn(in_features, hidden_size, num_layers, policy_rnn_dropout_p)
        self.mean_fc = nn.Linear(hidden_size, self.out_features * self.n_dist)
        self.log_scale_fc = nn.Linear(hidden_size, self.out_features * self.n_dist)
        self.prob_fc = nn.Linear(hidden_size, self.out_features * self.n_dist)
        self.register_buffer("one_hot_embedding_eye", torch.eye(self.n_dist))
        self.register_buffer("ones", torch.ones(1, 1, self.n_dist))
        self._setup_action_bounds("", act_max_bound, act_min_bound, False)

        self.one_hot_embedding_eye: torch.Tensor = self.one_hot_embedding_eye
        self.action_max_bound: torch.Tensor = self.action_max_bound
        self.action_min_bound: torch.Tensor = self.action_min_bound
        if self.discrete_gripper:
            self.gripper_bounds: torch.Tensor = self.gripper_bounds
            self.gripper_fc = nn.Linear(hidden_size, 2)
            self.criterion = nn.CrossEntropyLoss()
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
        logit_probs, log_scales, means, gripper_act, _ = self(
            latent_plan, perceptual_emb, latent_goal
        )
        pred_actions = self._sample(logit_probs, log_scales, means, gripper_act)
        loss = self._loss(logit_probs, log_scales, means, gripper_act, actions)
        return loss, pred_actions

    def act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logit_probs, log_scales, means, gripper_act, self.hidden_state = self(
            latent_plan, perceptual_emb, latent_goal, self.hidden_state
        )
        pred_actions = self._sample(logit_probs, log_scales, means, gripper_act)
        return pred_actions

    def loss(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        actions: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logit_probs, log_scales, means, gripper_act, _ = self(
            latent_plan, perceptual_emb, latent_goal
        )
        return self._loss(logit_probs, log_scales, means, gripper_act, actions)

    def _loss(
        self,
        logit_probs: torch.Tensor,
        log_scales: torch.Tensor,
        means: torch.Tensor,
        gripper_act: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.discrete_gripper:
            logistics_loss = self._logistic_loss(
                logit_probs, log_scales, means, actions[:, :, :-1]
            )
            gripper_gt = actions[:, :, -1].clone()
            m = gripper_gt == -1
            gripper_gt[m] = 0
            gripper_act_loss = self.criterion(
                gripper_act.view(-1, 2), gripper_gt.view(-1).long()
            )
            total_loss = logistics_loss + self.gripper_alpha * gripper_act_loss
            return total_loss
        else:
            logistics_loss = self._logistic_loss(
                logit_probs, log_scales, means, actions
            )
            return logistics_loss

    def _setup_action_bounds(
        self, dataset_dir, act_max_bound, act_min_bound, load_action_bounds
    ):
        if load_action_bounds:
            try:
                statistics_path = (
                    Path(tacorl.__file__).parent
                    / dataset_dir
                    / "training/statistics.yaml"
                )
                statistics = OmegaConf.load(statistics_path)
                act_max_bound = statistics.act_max_bound
                act_min_bound = statistics.act_min_bound
                logger.info(f"Loaded action bounds from {statistics_path}")
            except FileNotFoundError:
                logger.info(
                    f"Could not load statistics.yaml in {statistics_path}, "
                    "taking action bounds defined in hydra conf"
                )
        if self.discrete_gripper:
            self.register_buffer(
                "gripper_bounds", torch.Tensor([act_min_bound[-1], act_max_bound[-1]])
            )
            act_max_bound = act_max_bound[:-1]  # for discrete grasp
            act_min_bound = act_min_bound[:-1]
        action_max_bound = torch.Tensor(act_max_bound).float()
        action_min_bound = torch.Tensor(act_min_bound).float()
        assert action_max_bound.shape[0] == self.out_features
        assert action_min_bound.shape[0] == self.out_features
        action_max_bound = action_max_bound.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, action_space]
        action_min_bound = action_min_bound.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, action_space]
        action_max_bound = (
            action_max_bound.unsqueeze(-1) * self.ones
        )  # broadcast to [1, 1, action_space, N_DIST]
        action_min_bound = (
            action_min_bound.unsqueeze(-1) * self.ones
        )  # broadcast to [1, 1, action_space, N_DIST]
        self.register_buffer("action_max_bound", action_max_bound)
        self.register_buffer("action_min_bound", action_min_bound)

    def _logistic_loss(
        self,
        logit_probs: torch.Tensor,
        log_scales: torch.Tensor,
        means: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Appropriate scale
        log_scales = torch.clamp(log_scales, min=LOG_SIG_MIN)
        # Broadcast actions (B, A, N_DIST)
        actions = actions.unsqueeze(-1) * self.ones
        # Approximation of CDF derivative (PDF)
        centered_actions = actions - means
        inv_stdv = torch.exp(-log_scales)
        assert torch.is_tensor(self.action_max_bound)
        assert torch.is_tensor(self.action_min_bound)
        act_range = (self.action_max_bound - self.action_min_bound) / 2.0
        plus_in = inv_stdv * (centered_actions + act_range / (self.num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_actions - act_range / (self.num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # Corner Cases
        log_cdf_plus = plus_in - F.softplus(
            plus_in
        )  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(
            min_in
        )  # log probability for edge case of 255 (before scaling)
        # Log probability in the center of the bin
        mid_in = inv_stdv * centered_actions
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        # Probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        # Log probability
        log_probs = torch.where(
            actions < self.action_min_bound + 1e-3,
            log_cdf_plus,
            torch.where(
                actions > self.action_max_bound - 1e-3,
                log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                    log_pdf_mid - np.log((self.num_classes - 1) / 2),
                ),
            ),
        )
        log_probs = log_probs + F.log_softmax(logit_probs, dim=-1)
        loss = -torch.sum(log_sum_exp(log_probs), dim=-1).mean()
        return loss

    # Sampling from logistic distribution
    def _sample(
        self,
        logit_probs: torch.Tensor,
        log_scales: torch.Tensor,
        means: torch.Tensor,
        gripper_act: torch.Tensor,
    ) -> torch.Tensor:
        # Selecting Logistic distribution (Gumbel Sample)
        r1, r2 = 1e-5, 1.0 - 1e-5
        temp = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        temp = logit_probs - torch.log(-torch.log(temp))
        argmax = torch.argmax(temp, -1)
        # TODO: find out why mypy complains about type
        dist = self.one_hot_embedding_eye[argmax]

        # Select scales and means
        log_scales = (dist * log_scales).sum(dim=-1)
        means = (dist * means).sum(dim=-1)

        # Inversion sampling for logistic mixture sampling
        scales = torch.exp(log_scales)  # Make positive
        u = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        actions = means + scales * (torch.log(u) - torch.log(1.0 - u))
        if self.discrete_gripper:
            gripper_cmd = self.gripper_bounds[gripper_act.argmax(dim=-1)]
            full_action = torch.cat([actions, gripper_cmd.unsqueeze(-1)], 2)
            return full_action
        else:
            return actions

    def forward(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # perceptual_emb = perceptual_emb[..., slice(*self.perceptual_emb_slice)]
        batch_size, seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        latent_plan = latent_plan.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([latent_plan, perceptual_emb], dim=-1)
        if self.include_goal:
            latent_goal = latent_goal.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, latent_goal], dim=-1)
        # b, s, (plan + visuo-propio + goal)
        if not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase):
            x, h_n = self.rnn(x, h_0)
        else:
            x = self.rnn(x)
            h_n = None
        probs = self.prob_fc(x)
        means = self.mean_fc(x)
        log_scales = self.log_scale_fc(x)
        log_scales = torch.clamp(log_scales, min=LOG_SIG_MIN)
        gripper_act = self.gripper_fc(x) if self.discrete_gripper else None
        # Appropriate dimensions
        logit_probs = probs.view(batch_size, seq_len, self.out_features, self.n_dist)
        means = means.view(batch_size, seq_len, self.out_features, self.n_dist)
        log_scales = log_scales.view(
            batch_size, seq_len, self.out_features, self.n_dist
        )
        return logit_probs, log_scales, means, gripper_act, h_n
