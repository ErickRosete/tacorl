import math

import torch
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    RelaxedOneHotCategorical,
)

from tacorl.utils.misc import atanh


class GumbelSoftmax(RelaxedOneHotCategorical):
    """
    A differentiable Categorical distribution using reparametrization trick with
    Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its
    log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    """

    def sample(self, sample_shape=torch.Size()):
        """Gumbel-softmax sampling. Note rsample is inherited from
        RelaxedOneHotCategorical"""
        shape = self._extended_shape(sample_shape)
        u = torch.empty(
            self.logits.expand(shape).size(),
            device=self.logits.device,
            dtype=self.logits.dtype,
        ).uniform_(0, 1)
        noisy_logits = self.logits.expand(shape) - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size(), hard: bool = False):
        # Reparameterization trick
        y_soft = super().rsample(sample_shape)
        if hard:
            # Straight through trick
            index = torch.argmax(y_soft, dim=-1)
            y_hard = F.one_hot(index, self.logits.shape[-1]).float()
            return (y_hard - y_soft).detach() + y_soft
        return y_soft

    def log_prob(self, value):
        """value is one-hot or relaxed"""
        if value.ndim == 0 or value.shape[-1] != self.logits.shape[-1]:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert (
                value.shape[-1] == self.logits.shape[-1]
            ), "Last dimension should match after one hot encoding"
        log_pi = -value * F.log_softmax(self.logits, dim=-1)
        return -torch.sum(log_pi, dim=-1, keepdim=True)


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Independent(Normal(normal_mean, normal_std), 1)

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample((n,))

        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        correction term is mathematically equivalent to - log(1 - tanh(x)^2).
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = -2 * (
            math.log(2) - pre_tanh_value - F.softplus(-2 * pre_tanh_value)
        ).sum(dim=-1)
        return (log_prob + correction).unsqueeze(-1)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999, 0.999)
            pre_tanh_value = atanh(value)
        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self):
        """
        Sampling in the reparameterization case.
        """
        z = self.normal.rsample()
        z.requires_grad_()
        return torch.tanh(z), z

    def rsample(self):
        """
        Sampling in the reparameterization case.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value

    def sample(self):
        pre_tanh_value = self.normal.sample()
        value = torch.tanh(pre_tanh_value)
        return value.detach()

    def sample_and_logprob(self):
        pre_tanh_value = self.normal.sample()
        value = torch.tanh(pre_tanh_value)
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_logprob_and_pretanh(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p, pre_tanh_value

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    @property
    def stddev(self):
        return self.normal_std
