from copy import copy
from typing import Iterable
import torch
from torch import nn
from torch.distributions import Uniform, Normal
import numpy as np
from modules.util import (
    reparameterize,
    kldiv,
    SNR,
    Etheta,
    Phi,
    LogU_cdf,
    LogN_cdf
)


REGISTERED_PRUNING_LAYERS = []


class LogUniformPruningLayer(nn.Module):

    @classmethod
    def register(cls):
        REGISTERED_PRUNING_LAYERS.append(cls)

    # The defaults are the ones used in the paper
    def __init__(
            self,
            d,
            a=-20,
            b=0,
            n_samples=1,
            eps=1e-5,
            kl_averaging='sum',
            thresh=1.,
            enabled=True,
            axis=-1
    ):
        """
        Base BMRS pruning layer which uses SNR for pruning
        :param d: The input dimensionality for the pruning layer (pad with 1s for broadcasting)
        :param a: The (log) left bound of truncation
        :param b: The (log) right bound of truncation
        :param n_samples: The number of samples to use for SCI
        :param eps: A small epsilon for numerical stability
        :param kl_averaging: 'sum' to sum the KL divergence loss, 'mean' to use the mean
        :param thresh: The pruning threshold to use
        :param enabled: Enable or disable the pruning layer
        :param axis: Which axis the multiplicative noise is applied to
        """
        super().__init__()
        valid_averaging = ['sum', 'mean']
        assert kl_averaging in valid_averaging, f"Invalid averaging value, valid terms are: {valid_averaging}"
        self.kl_averaging = kl_averaging
        self.a = a
        self.b = b
        # Using some hand-coded values from the paper for stability
        self.logsigmin = -20
        self.logsigmax = 5
        self.thresh = thresh

        if isinstance(d, Iterable):
            self.d = d
        else:
            self.d = [d]
        if axis != -1:
            for dim in range(len(self.d)):
                if dim != axis:
                    self.d[dim] = 1
        self.axis = axis
        self.eps = eps
        # Not used for now, in theory could do more Monte Carlo samples
        self.n_samples = n_samples

        self.umin = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.umax = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        # Store the variational parameters $\mu$ and $\sigma$
        self.mu = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.logsigma = nn.Parameter(torch.ones(self.d) * -5, requires_grad=True)
        self.mask = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.etheta_cache = None

        self.enabled = enabled

    def reparameterize(self, size):
        """
        Reparameterization trick from Neklyudov et al. 2017
        :param size: The size of the input
        :return: Samples from the variational approximation
        """
        # Sample y_{i} from a uniform distribution
        uniform = Uniform(self.umin, self.umax)
        y = uniform.sample(size).squeeze()
        mu = self.mu.unsqueeze(0).expand(size)
        sigma = torch.clamp(self.logsigma.unsqueeze(0).expand(size), min=self.logsigmin, max=self.logsigmax).exp()

        return reparameterize(mu, sigma, self.a, self.b, y, self.eps)


    def kl(self, averaging='sum'):
        """
        Calculates the KL divergence term between LogN approximation and LogU prior based on derivation in
        Neklyudov et al. 2017
        """
        if not self.enabled:
            return 0.
        mu = self.mu
        logsigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax)
        sigma = logsigma.exp()

        total = kldiv(mu, sigma, self.a, self.b, self.eps)

        if averaging == 'sum':
            return total.sum()
        elif averaging == 'mean':
            return total.mean()

    def SNR(self):
        """
        Calculate the SNR to use as a threshold for dropout
        """
        mu = self.mu
        sigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax).exp()

        return SNR(mu, sigma, self.a, self.b, self.eps)

    def Etheta(self):
        """
        Calculates the expected value of each theta (used for test time)
        """
        mu = self.mu
        sigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax).exp()
        return Etheta(mu, sigma, self.a, self.b, self.eps)

    def compress(self, set_mask=True, thresh=None, **kwargs):
        """
        Perform compression and return a mask for the pruning layer
        :param set_mask: Whether or not to remove a parameters
        :param thresh: A threshold to use for compression; if None, uses the layer default
        :param kwargs:
        :return: A mask over the multiplicative noise indicating which variables are removed
        """
        if thresh == None:
            thresh = self.thresh
        mask = (self.SNR() > thresh).float()
        if set_mask:
            self.mask.data = mask
        return 1 - mask

    def forward(self, x):
        # Sample from the variational distribution
        if self.axis == -1:
            assert list(x.shape[1:]) == self.d
        else:
            assert x.shape[1:][self.axis] == self.d[self.axis]

        if self.enabled:
            if self.training:
                self.etheta_cache = None
                theta = self.reparameterize(x.shape) * self.mask
            else:
                if self.etheta_cache is not None:
                    etheta = self.etheta_cache
                else:
                    etheta = self.Etheta()
                    self.etheta_cache = etheta
                params = etheta * self.mask
                theta = params.unsqueeze(0).expand(x.shape)

            # Multiplicative noise
            return x * theta
        else:
            # Pass through when disabled
            return x

    def __str__(self):
        return "SNR"


class LogUniformL2NormPruningLayer(nn.Module):

    @classmethod
    def register(cls):
        REGISTERED_PRUNING_LAYERS.append(cls)

    # The defaults are the ones used in the paper
    def __init__(
            self,
            d,
            a=-20,
            b=0,
            n_samples=1,
            eps=1e-5,
            kl_averaging='sum',
            thresh=0.5,
            enabled=True,
            axis=-1
    ):
        """
        Base BMRS pruning layer which uses the L2 norm of weight matrices for pruning
        :param d: The input dimensionality for the pruning layer (pad with 1s for broadcasting)
        :param a: The (log) left bound of truncation
        :param b: The (log) right bound of truncation
        :param n_samples: The number of samples to use for SCI
        :param eps: A small epsilon for numerical stability
        :param kl_averaging: 'sum' to sum the KL divergence loss, 'mean' to use the mean
        :param thresh: The pruning threshold to use
        :param enabled: Enable or disable the pruning layer
        :param axis: Which axis the multiplicative noise is applied to
        """
        super().__init__()
        valid_averaging = ['sum', 'mean']
        assert kl_averaging in valid_averaging, f"Invalid averaging value, valid terms are: {valid_averaging}"
        self.kl_averaging = kl_averaging
        self.a = a
        self.b = b
        # Using some hand-coded values from the paper for stability
        self.logsigmin = -20
        self.logsigmax = 5
        self.thresh = thresh

        if isinstance(d, Iterable):
            self.d = d
        else:
            self.d = [d]
        if axis != -1:
            for dim in range(len(self.d)):
                if dim != axis:
                    self.d[dim] = 1
        self.axis = axis
        self.eps = eps
        # Not used for now, in theory could do more Monte Carlo samples
        self.n_samples = n_samples

        self.umin = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.umax = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        # Store the variational parameters $\mu$ and $\sigma$
        self.mu = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.logsigma = nn.Parameter(torch.ones(self.d) * -5, requires_grad=True)
        self.mask = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.etheta_cache = None

        self.enabled = enabled

    def reparameterize(self, size):
        """
        Reparameterization trick from Neklyudov et al. 2017
        :param size: The size of the input
        :return: Samples from the variational approximation
        """
        # Sample y_{i} from a uniform distribution
        uniform = Uniform(self.umin, self.umax)
        y = uniform.sample(size).squeeze()
        mu = self.mu.unsqueeze(0).expand(size)
        sigma = torch.clamp(self.logsigma.unsqueeze(0).expand(size), min=self.logsigmin, max=self.logsigmax).exp()

        return reparameterize(mu, sigma, self.a, self.b, y, self.eps)


    def kl(self, averaging='sum'):
        """
        Calculates the KL divergence term between LogN approximation and LogU prior
        """
        if not self.enabled:
            return 0.
        mu = self.mu
        logsigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax)
        sigma = logsigma.exp()

        total = kldiv(mu, sigma, self.a, self.b, self.eps)

        if averaging == 'sum':
            return total.sum()
        elif averaging == 'mean':
            return total.mean()

    def Etheta(self):
        """
        Calculates the expected value of each theta (used for test time)
        """
        mu = self.mu
        sigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax).exp()
        return Etheta(mu, sigma, self.a, self.b, self.eps)

    def compress(self, set_mask=True, thresh=None, l2_norm=None, **kwargs):
        """
        Perform compression based on the L2 norm.
        :param set_mask: Set to true to remove the compressed neurons
        :param thresh: A pruning threshold for the L2 norm
        :param l2_norm: L2 norm of the input weights to each noise variable. If None, uses the expected value of theta
        :param kwargs:
        :return: A mask over the multiplicative noise indicating which variables are removed
        """
        l2_norm = l2_norm.reshape(self.mask.shape)
        if thresh == None:
            thresh = self.thresh
        if l2_norm == None:
            mask = (self.Etheta() > thresh).float()
        else:
            mask = (l2_norm > thresh).float()
        if set_mask:
            self.mask.data = mask
        return 1 - mask

    def forward(self, x):
        # Sample from the variational distribution
        if self.axis == -1:
            assert list(x.shape[1:]) == self.d
        else:
            assert x.shape[1:][self.axis] == self.d[self.axis]

        if self.enabled:
            if self.training:
                self.etheta_cache = None
                theta = self.reparameterize(x.shape) * self.mask
            else:
                if self.etheta_cache is not None:
                    etheta = self.etheta_cache
                else:
                    etheta = self.Etheta()
                    self.etheta_cache = etheta
                params = etheta * self.mask
                theta = params.unsqueeze(0).expand(x.shape)

            # Multiplicative noise
            return x * theta
        else:
            # Pass through when disabled
            return x

    def __str__(self):
        return "L2"


class LogUniformApproximateDiracBMRPruningLayer(nn.Module):

    @classmethod
    def register(cls):
        REGISTERED_PRUNING_LAYERS.append(cls)

    # The defaults are the ones used in the paper
    def __init__(
            self,
            d,
            a=-20,
            b=0,
            n_samples=1,
            eps=1e-5,
            kl_averaging='sum',
            thresh=0.,
            enabled=True,
            pruning_eps=None,
            axis=-1
    ):
        """
        BMRS_N pruning layer which uses a truncated log-normal distribution as the reduced prior
        :param d: The input dimensionality for the pruning layer (pad with 1s for broadcasting)
        :param a: The (log) left bound of truncation
        :param b: The (log) right bound of truncation
        :param n_samples: The number of samples to use for SCI
        :param eps: A small epsilon for numerical stability
        :param kl_averaging: 'sum' to sum the KL divergence loss, 'mean' to use the mean
        :param thresh: The pruning threshold to use (should be 0 but can be changed)
        :param enabled: Enable or disable the pruning layer
        :param pruning_eps: A small value around 0 to use for sigma_p_tilde
        :param axis: Which axis the multiplicative noise is applied to
        """
        super().__init__()
        valid_averaging = ['sum', 'mean']
        assert kl_averaging in valid_averaging, f"Invalid averaging value, valid terms are: {valid_averaging}"
        self.kl_averaging = kl_averaging
        self.a = a
        self.b = b
        # Using some hand-coded values from the paper for stability
        self.logsigmin = -20
        self.logsigmax = 5
        self.thresh = thresh

        if isinstance(d, Iterable):
            self.d = d
        else:
            self.d = [d]
        if axis != -1:
            for dim in range(len(self.d)):
                if dim != axis:
                    self.d[dim] = 1
        self.axis = axis
        self.eps = eps
        # Not used for now, in theory could do more Monte Carlo samples
        self.n_samples = n_samples

        self.umin = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.umax = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        # Store the variational parameters $\mu$ and $\sigma$
        self.mu = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.logsigma = nn.Parameter(torch.ones(self.d) * -5, requires_grad=True)
        self.mask = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.etheta_cache = None

        if pruning_eps is None:
            self.pruning_eps = nn.Parameter(torch.tensor(np.sqrt(1e-12)).double(), requires_grad=False)

        else:
            self.pruning_eps = nn.Parameter(torch.tensor(pruning_eps), requires_grad=False)

        self.enabled = enabled


    def reparameterize(self, size):
        """
        Reparameterization trick from Neklyudov et al. 2017
        :param size: The size of the input
        :return: Samples from the variational approximation
        """
        uniform = Uniform(self.umin, self.umax)
        y = uniform.sample(size).squeeze()
        mu = self.mu.unsqueeze(0).expand(size)
        sigma = torch.clamp(self.logsigma.unsqueeze(0).expand(size), min=self.logsigmin, max=self.logsigmax).exp()

        return reparameterize(mu, sigma, self.a, self.b, y, self.eps)

    def kl(self, averaging='sum'):
        """
        Calculates the KL divergence term between LogN approximation and LogU prior
        """
        mu = self.mu
        logsigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax)
        sigma = logsigma.exp()

        total = kldiv(mu, sigma, self.a, self.b, self.eps)

        if averaging == 'sum':
            return total.sum()
        elif averaging == 'mean':
            return total.mean()

    def Etheta(self, mu=None, sigma=None):
        """
        Calculates the expected value of each theta (used for test time)
        """
        if mu == None:
            mu = self.mu
            sigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax).exp()
        return Etheta(mu, sigma, self.a, self.b, self.eps)

    def qtilde(
            self,
            Q,
            sigma_p_tilde=None,
            mu_p_tilde=None
    ):
        """
        Determine Q_tilde based on Q, P, and P_tilde
        :param Q: The parameters of this layer (mu and var)
        :param sigma_p_tilde: Variance for the reduced prior (something close to 0)
        :param mu_p_tilde: Mean of the reduced prior
        :return: The variational approximation under the reduced prior
        """
        if sigma_p_tilde == None:
            sigma_p_tilde = self.pruning_eps

        mu_pt = self.a if mu_p_tilde == None else mu_p_tilde

        # Determine the new variational parameters
        var_q_tild = 1 / ( 1/ Q[1] + 1 / sigma_p_tilde**2)
        mu_q_tild = var_q_tild * (Q[0] / Q[1] + mu_pt / sigma_p_tilde**2)

        return (mu_q_tild, var_q_tild)

    def deltaF(self, Q, Q_tilde, sigma_p_tilde=None, mu_p_tilde=None):
        """
        Determines $$\Delta F$$
        :param Q: The parameters of this layer (mu and var)
        :param Q_tilde: The variational approximation under the reduced prior
        :param sigma_p_tilde: Variance for the reduced prior (something close to 0)
        :param mu_p_tilde: Mean of the reduced prior
        :return: The change in variational free energy under the reduced prior
        """
        d = 1. if sigma_p_tilde == None else 2.
        if sigma_p_tilde == None:
            sigma_p_tilde = self.pruning_eps

        mu_pt = self.a if mu_p_tilde == None else mu_p_tilde

        alpha_q = (self.a - Q[0]) / torch.sqrt(Q[1])
        beta_q = (self.b - Q[0]) / torch.sqrt(Q[1])
        Z_q = Phi(beta_q) - Phi(alpha_q)

        alpha_pt = (self.a - mu_pt) / sigma_p_tilde
        beta_pt = (self.b - mu_pt) / sigma_p_tilde
        Z_pt = Phi(beta_pt) - Phi(alpha_pt)

        alpha_qt = (self.a - Q_tilde[0]) / torch.sqrt(Q_tilde[1])
        beta_qt = (self.b - Q_tilde[0]) / torch.sqrt(Q_tilde[1])
        Z_qt = Phi(beta_qt) - Phi(alpha_qt)

        # b and a already log
        term1 = torch.log((Z_qt * (self.b - self.a)) / (Z_q * Z_pt))

        term2 = 0.5 * torch.log((Q_tilde[1]) / (Q[1] * sigma_p_tilde ** 2))

        term4 = (mu_pt ** 2) / sigma_p_tilde ** 2 - (Q_tilde[0] ** 2) / Q_tilde[1]

        term3 = -0.5 * ((Q[0] ** 2) / Q[1] + term4)
        return term1 + term2 + term3


    def bmr(self, sigma_p_tilde=None, mu_p_tilde=None):
        """
        Performs the BMR calculation
        :param sigma_p_tilde: Variance for the reduced prior (something close to 0)
        :param mu_p_tilde: Mean of the reduced prior
        :return: The change in variational free energy under the reduced prior
        """

        Q = (self.mu.double().cuda(), torch.clamp(self.logsigma.double().cuda(), min=self.logsigmin, max=self.logsigmax).exp()**2)

        # Calculate Q_tilde
        Q_tilde = self.qtilde(Q, sigma_p_tilde=sigma_p_tilde, mu_p_tilde=mu_p_tilde)
        dF = self.deltaF(Q, Q_tilde, sigma_p_tilde=sigma_p_tilde, mu_p_tilde=mu_p_tilde)

        return dF

    def compress(self, set_mask=True, sigma_p_tilde=None, mu_p_tilde=None, thresh=None, **kwargs):
        """
        Perform compression based on $$\Delta F$$ for an approximate dirac spike at 0.
        :param set_mask: Set to true to remove the compressed neurons
        :param sigma_p_tilde: Variance for the reduced prior (something close to 0)
        :param mu_p_tilde: Mean of the reduced prior
        :param thresh: A pruning threshold (should be 0 but can be changed)
        :param kwargs:
        :return: A mask over the multiplicative noise indicating which variables are removed
        """
        if thresh == None:
            thresh = self.thresh
        dF = self.bmr(sigma_p_tilde=sigma_p_tilde, mu_p_tilde=mu_p_tilde)
        mask = (dF < thresh).float()
        # print(f"BMR: {1 - mask.mean()}")
        # print(f"SNR: {1 - (self.SNR() > 1.0).float().mean()}")
        if set_mask:
            self.mask.data = mask
        return 1 - mask

    def forward(self, x):
        # Sample from the variational distribution
        if self.axis == -1:
            assert list(x.shape[1:]) == self.d
        else:
            assert x.shape[1:][self.axis] == self.d[self.axis]

        if self.enabled:
            if self.training:
                self.etheta_cache = None
                theta = self.reparameterize(x.shape) * self.mask
            else:
                if self.etheta_cache is not None:
                    etheta = self.etheta_cache
                else:
                    etheta = self.Etheta()
                    self.etheta_cache = etheta
                params = etheta * self.mask
                theta = params.unsqueeze(0).expand(x.shape)

            # Multiplicative noise
            return x * theta
        else:
            # Pass through when disabled
            return x

    def __str__(self):
        return "BMR Approximate Spike"


class LogUniformCDFBMRPruningLayer(nn.Module):

    @classmethod
    def register(cls):
        REGISTERED_PRUNING_LAYERS.append(cls)

    # The defaults are the ones used in the paper
    def __init__(
            self,
            d,
            a=-20,
            b=0,
            n_samples=1,
            eps=1e-5,
            kl_averaging='sum',
            thresh=0.,
            enabled=True,
            axis=-1
    ):
        """
        BMRS_U pruning layer which uses a reduced truncated log-uniform distribution as the reduced prior
        :param d: The input dimensionality for the pruning layer (pad with 1s for broadcasting)
        :param a: The (log) left bound of truncation
        :param b: The (log) right bound of truncation
        :param n_samples: The number of samples to use for SCI
        :param eps: A small epsilon for numerical stability
        :param kl_averaging: 'sum' to sum the KL divergence loss, 'mean' to use the mean
        :param thresh: The pruning threshold to use (should be 0 but can be changed)
        :param enabled: Enable or disable the pruning layer
        :param axis: Which axis the multiplicative noise is applied to
        """
        super().__init__()
        valid_averaging = ['sum', 'mean']
        assert kl_averaging in valid_averaging, f"Invalid averaging value, valid terms are: {valid_averaging}"
        self.kl_averaging = kl_averaging
        self.a = a
        self.b = b
        # Using some hand-coded values from the paper for stability
        self.logsigmin = -20
        self.logsigmax = 5
        self.thresh = thresh

        if isinstance(d, Iterable):
            self.d = d
        else:
            self.d = [d]
        if axis != -1:
            for dim in range(len(self.d)):
                if dim != axis:
                    self.d[dim] = 1
        self.axis = axis
        self.eps = eps
        # Not used for now, in theory could do more Monte Carlo samples
        self.n_samples = n_samples

        self.umin = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.umax = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        # Store the variational parameters $\mu$ and $\sigma$
        self.mu = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.logsigma = nn.Parameter(torch.ones(self.d) * -5, requires_grad=True)
        self.mask = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.etheta_cache = None

        self.enabled = enabled


    def reparameterize(self, size):
        """
        Reparameterization trick from Neklyudov et al. 2017
        :param size: The size of the input
        :return: Samples from the variational approximation
        """
        # Sample y_{i} from a uniform distribution
        uniform = Uniform(self.umin, self.umax)
        y = uniform.sample(size).squeeze()
        mu = self.mu.unsqueeze(0).expand(size)
        sigma = torch.clamp(self.logsigma.unsqueeze(0).expand(size), min=self.logsigmin, max=self.logsigmax).exp()

        return reparameterize(mu, sigma, self.a, self.b, y, self.eps)

    def kl(self, averaging='sum'):
        """
        Calculates the KL divergence term between LogN approximation and LogU prior
        """
        mu = self.mu
        logsigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax)
        sigma = logsigma.exp()

        total = kldiv(mu, sigma, self.a, self.b, self.eps)

        if averaging == 'sum':
            return total.sum()
        elif averaging == 'mean':
            return total.mean()

    def Etheta(self, mu=None, sigma=None):
        """
        Calculates the expected value of each theta (used for test time)
        """
        if mu == None:
            mu = self.mu
            sigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax).exp()
        return Etheta(mu, sigma, self.a, self.b, self.eps)


    def bmr(self, mu_p_tilde=None):
        """
        Calculate $$\Delta F$$ based on a reduced truncated log-uniform prior
        :param mu_p_tilde: The bit precision to use for the right truncation bound i.e. $$b' = \frac{1}{2^{mu_p_tilde}}$$
        :return: The change in variational free energy under the reduced prior
        """
        if mu_p_tilde == None:
            mu_p_tilde = torch.tensor(8).cuda().double()

        Q = (torch.clamp(self.mu, min=self.a, max=self.b).double().cuda(),
             torch.clamp(self.logsigma.double().cuda(), min=self.logsigmin, max=None).exp())

        ap = torch.tensor(np.log(1 / (2**23))).cuda()
        bp = torch.log(1 / (2**mu_p_tilde))
        q_pdf = torch.log(LogN_cdf(ap, bp, Q[0], Q[1], self.a, self.b))
        u_pdf = torch.log(LogU_cdf(ap, bp, self.a, self.b))

        dF = ((q_pdf - u_pdf) >= 0.0).float() - 0.5
        return dF

    def compress(self, set_mask=True, mu_p_tilde=None, thresh=None, **kwargs):
        """
        Perform compression based on $$\Delta F$$ for a reduced truncated log-uniform distribution.
        :param set_mask: Set to true to remove the compressed neurons
        :param mu_p_tilde: The bit precision to use for the right truncation bound i.e. $$b' = \frac{1}{2^{mu_p_tilde}}$$
        :param thresh: A pruning threshold (should be 0 but can be changed)
        :param kwargs:
        :return: A mask over the multiplicative noise indicating which variables are removed
        """
        if thresh == None:
            thresh = self.thresh

        dF = self.bmr(mu_p_tilde=mu_p_tilde)
        mask = (dF < thresh).float()

        if set_mask:
            self.mask.data = mask
        return 1 - mask

    def forward(self, x):
        # Sample from the variational distribution
        if self.axis == -1:
            assert list(x.shape[1:]) == self.d
        else:
            assert x.shape[1:][self.axis] == self.d[self.axis]

        if self.enabled:
            if self.training:
                self.etheta_cache = None
                theta = self.reparameterize(x.shape) * self.mask
            else:
                if self.etheta_cache is not None:
                    etheta = self.etheta_cache
                else:
                    etheta = self.Etheta()
                    self.etheta_cache = etheta
                params = etheta * self.mask
                theta = params.unsqueeze(0).expand(x.shape)

            # Multiplicative noise
            return x * theta
        else:
            # Pass through when disabled
            return x

    def __str__(self):
        return "BMR Reduced Log Uniform"


class LogUniformAllBMRPruningLayer(nn.Module):

    @classmethod
    def register(cls):
        REGISTERED_PRUNING_LAYERS.append(cls)

    # The defaults are the ones used in the paper
    def __init__(
            self,
            d,
            a=-20,
            b=0,
            n_samples=1,
            eps=1e-5,
            kl_averaging='sum',
            thresh=0.,
            enabled=True,
            pruning_eps=None,
            axis=-1
    ):
        """
        Convenience class combining all of the pruning methods for generating the post-training pruning plots (see
        individual classes for details on each method)
        :param d:
        :param a:
        :param b:
        :param n_samples:
        :param eps:
        :param kl_averaging:
        :param thresh:
        :param enabled:
        :param pruning_eps:
        :param axis:
        """
        super().__init__()
        valid_averaging = ['sum', 'mean']
        assert kl_averaging in valid_averaging, f"Invalid averaging value, valid terms are: {valid_averaging}"
        self.kl_averaging = kl_averaging
        self.a = a
        self.b = b
        # Using some hand-coded values from the paper for stability
        self.logsigmin = -20
        self.logsigmax = 5
        self.thresh = thresh

        if isinstance(d, Iterable):
            self.d = d
        else:
            self.d = [d]
        if axis != -1:
            for dim in range(len(self.d)):
                if dim != axis:
                    self.d[dim] = 1
        self.axis = axis
        self.eps = eps
        # Not used for now, in theory could do more Monte Carlo samples
        self.n_samples = n_samples

        self.umin = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.umax = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        # Store the variational parameters $\mu$ and $\sigma$
        self.mu = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.logsigma = nn.Parameter(torch.ones(self.d) * -5, requires_grad=True)
        self.mask = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.etheta_cache = None

        if pruning_eps is None:

            self.pruning_eps = nn.Parameter(torch.tensor(np.sqrt(1e-12)).double(), requires_grad=False)

        else:
            self.pruning_eps = nn.Parameter(torch.tensor(pruning_eps), requires_grad=False)

        self.enabled = enabled
        self.thresh = 0

    def reparameterize(self, size):
        # Sample y_{i} from a uniform distribution
        uniform = Uniform(self.umin, self.umax)
        y = uniform.sample(size).squeeze()
        mu = self.mu.unsqueeze(0).expand(size)
        sigma = torch.clamp(self.logsigma.unsqueeze(0).expand(size), min=self.logsigmin, max=self.logsigmax).exp()

        return reparameterize(mu, sigma, self.a, self.b, y, self.eps)

    def kl(self, averaging='sum'):
        """
        Calculates the KL divergence term between LogN approximation and LogU prior
        """
        mu = self.mu
        logsigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax)
        sigma = logsigma.exp()

        total = kldiv(mu, sigma, self.a, self.b, self.eps)

        if averaging == 'sum':
            return total.sum()
        elif averaging == 'mean':
            return total.mean()

    def SNR(self, mu=None, sigma=None):
        """
        Calculate the SNR to use as a threshold for dropout
        """
        if mu == None:
            mu = self.mu
            sigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax).exp()

        return SNR(mu, sigma, self.a, self.b, self.eps)

    def Etheta(self, mu=None, sigma=None):
        """
        Calculates the expected value of each theta (used for test time)
        """
        if mu == None:
            mu = self.mu
            sigma = torch.clamp(self.logsigma, min=self.logsigmin, max=self.logsigmax).exp()
        return Etheta(mu, sigma, self.a, self.b, self.eps)

    def qtilde(
            self,
            Q,
            sigma_p_tilde=None,
            mu_p_tilde=None
    ):
        if sigma_p_tilde == None:
            sigma_p_tilde = self.pruning_eps

        mu_pt = self.a if mu_p_tilde == None else mu_p_tilde

        # Determine the new variational parameters
        var_q_tild = 1 / ( 1/ Q[1] + 1 / sigma_p_tilde**2)
        mu_q_tild = var_q_tild * (Q[0] / Q[1] + mu_pt / sigma_p_tilde**2)

        return (mu_q_tild, var_q_tild)

    def deltaF(self, Q, Q_tilde, sigma_p_tilde=None, mu_p_tilde=None):
        d = 1. if sigma_p_tilde == None else 2.
        if sigma_p_tilde == None:
            sigma_p_tilde = self.pruning_eps

        mu_pt = self.a if mu_p_tilde == None else mu_p_tilde

        alpha_q = (self.a - Q[0]) / torch.sqrt(Q[1])
        beta_q = (self.b - Q[0]) / torch.sqrt(Q[1])
        Z_q = Phi(beta_q) - Phi(alpha_q)

        alpha_pt = (self.a - mu_pt) / sigma_p_tilde
        beta_pt = (self.b - mu_pt) / sigma_p_tilde
        Z_pt = Phi(beta_pt) - Phi(alpha_pt)

        alpha_qt = (self.a - Q_tilde[0]) / torch.sqrt(Q_tilde[1])
        beta_qt = (self.b - Q_tilde[0]) / torch.sqrt(Q_tilde[1])
        Z_qt = Phi(beta_qt) - Phi(alpha_qt)

        # b and a already log
        term1 = torch.log((Z_qt * (self.b - self.a)) / (Z_q * Z_pt))

        term2 = 0.5 * torch.log((Q_tilde[1]) / (Q[1] * sigma_p_tilde ** 2))
        # term2 = 0.5 * torch.log((Q_tilde[1]) / (Q[1] * sigma_p_tilde**2))

        term4 = (mu_pt ** 2) / sigma_p_tilde ** 2 - (Q_tilde[0] ** 2) / Q_tilde[1]

        term3 = -0.5 * ((Q[0] ** 2) / Q[1] + term4)
        return term1 + term2 + term3


    def bmr(self, type='bmr_approx', **kwargs):
        if type == 'bmr_approx':
            return self.bmr_approximate(**kwargs)
        elif type == 'bmr_cdf':
            return self.bmr_cdf(**kwargs)
        else:
            assert False, "Invalid prior selection"

    def bmr_approximate(self, sigma_p_tilde=None, mu_p_tilde=None):
        """
        Prunes neurons whos SNR is below the threshold
        """

        Q = (self.mu.double(), torch.clamp(self.logsigma.double(), min=self.logsigmin, max=self.logsigmax).exp()**2)

        # Calculate Q_tilde
        Q_tilde = self.qtilde(Q, sigma_p_tilde=sigma_p_tilde, mu_p_tilde=mu_p_tilde)
        dF = self.deltaF(Q, Q_tilde, sigma_p_tilde=sigma_p_tilde, mu_p_tilde=mu_p_tilde)

        return dF


    def bmr_cdf(self, sigma_p_tilde=None, mu_p_tilde=None):
        """
        Prunes neurons whos SNR is below the threshold
        """

        if mu_p_tilde == None:
            # mu_p_tilde = torch.tensor(self.a / 2).cuda().double()
            mu_p_tilde = torch.tensor(8).cuda().double()

        Q = (torch.clamp(self.mu, min=self.a, max=self.b).double().cuda(),
             torch.clamp(self.logsigma.double().cuda(), min=self.logsigmin, max=self.logsigmax).exp())


        ap = torch.tensor(np.log(1 / (2 ** 23))).cuda()
        bp = torch.log(1 / (2**mu_p_tilde))
        q_pdf = torch.log(LogN_cdf(ap, bp, Q[0], Q[1], self.a, self.b))
        u_pdf = torch.log(LogU_cdf(ap, bp, self.a, self.b))
        dF = q_pdf - u_pdf
        return dF


    def compress(self, set_mask=True, sigma_p_tilde=None, mu_p_tilde=None, thresh=None, **kwargs):
        if thresh == None:
            thresh = self.thresh

        dF = self.bmr(sigma_p_tilde=sigma_p_tilde, mu_p_tilde=mu_p_tilde)
        mask = (dF < thresh).float()

        if set_mask:
            self.mask.data = mask
        return 1 - mask

    def forward(self, x):
        # Sample from the variational distribution
        if self.axis == -1:
            assert list(x.shape[1:]) == self.d
        else:
            assert x.shape[1:][self.axis] == self.d[self.axis]

        if self.enabled:
            if self.training:
                self.etheta_cache = None
                theta = self.reparameterize(x.shape) * self.mask
            else:
                if self.etheta_cache is not None:
                    etheta = self.etheta_cache
                else:
                    etheta = self.Etheta()
                    self.etheta_cache = etheta
                params = etheta * self.mask
                theta = params.unsqueeze(0).expand(x.shape)

            # Multiplicative noise
            return x * theta
        else:
            # Pass through when disabled
            return x

    def __str__(self):
        return "All Methods"


"""
Register everything
"""
LogUniformPruningLayer.register()
LogUniformL2NormPruningLayer.register()
LogUniformApproximateDiracBMRPruningLayer.register()
LogUniformCDFBMRPruningLayer.register()
LogUniformAllBMRPruningLayer.register()


"""
Utilities for working with modules that contain structured pruning layers
"""
def gather_pruning_layers(model):
    """
    Gets all of the pruning layers in a given model in a list
    :param model: A module with pruning layers
    :return: A list of all of the pruning layers in the model
    """
    pruning_layers = []
    for name, module in model.named_modules():
        if type(module) in REGISTERED_PRUNING_LAYERS:
            pruning_layers.append(module)
    return pruning_layers


def calculate_model_kl(model, averaging='sum'):
    """
    Calculates the KL divergence for the full model
    :param model: A module with pruning layers
    :param averaging: 'sum' to sum the KL divergence loss, 'mean' to use the mean
    :return: The KL divergence for the entire model
    """
    # Reconstruction loss
    pruning = gather_pruning_layers(model)
    kl_loss = 0.
    # KL loss
    for j,l in enumerate(pruning):
        kl = l.kl(averaging=averaging)
        kl_loss += kl
    if averaging == 'mean':
        kl_loss = kl_loss / len(pruning)
    return kl_loss


def get_weight_l2(model):
    """
    Gets the L2 norm of weight vectors which are inputs to pruning layers in a model
    :param model: A module with pruning layers
    :return: A list of the L2 norm or the weights which are inputs to each pruning neuron
    """
    l2 = []
    prev_layer = None
    for name, module in model.named_modules():
        if type(module) in REGISTERED_PRUNING_LAYERS:
            l2.append(torch.linalg.vector_norm(prev_layer.weight.data, dim=tuple(range(1,len(prev_layer.weight.data.shape)))))
        elif type(module) in [nn.Conv2d, nn.Linear]:
            prev_layer = module
    return l2


def disable(model):
    """
    Disables pruning
    :param model:
    :return:
    """
    for l in gather_pruning_layers(model):
        l.enabled = False


def enable(model):
    """
    Enables pruning
    :param model:
    :return:
    """
    for l in gather_pruning_layers(model):
        l.enabled = True


def usable_neuron_pct(model):
    """
    Calculate the number of inactive neurons in a given model
    :param model: A module with pruning layers
    :return: The percentage of inactive neurons in a model
    """
    pruning = gather_pruning_layers(model)
    masks = [1 - l.mask for l in pruning]
    usable = sum([m.sum() for m in masks])
    total = sum([m.view(-1).shape[0] for m in masks])
    return (usable / total).item() * 100


def compress(model, set_mask=True, sigma_p_tilde=None, mu_p_tilde=None, thresh=None):
    """
    Performs compression of a given model
    :param model: A module with pruning layers
    :param set_mask: Whether or not to remove neurons which are compressed
    :param sigma_p_tilde: Epsilon for the approximate dirac spike
    :param mu_p_tilde: Either the lower bound of truncation for the approximate dirac spike or the upper bit precision for the reduced log-uniform distribution
    :param thresh: A pruning threshold
    :return: The percentage of neurons which are removed from the model
    """
    l2_norms = get_weight_l2(model)
    pruning = gather_pruning_layers(model)
    masks = [l.compress(set_mask=set_mask, sigma_p_tilde=sigma_p_tilde, mu_p_tilde=mu_p_tilde, thresh=thresh, l2_norm=w) for l,w in zip(pruning, l2_norms) ]
    usable = sum([m.sum() for m in masks])
    total = sum([m.view(-1).shape[0] for m in masks])
    return (usable / total).item() * 100


def total_parameters(model):
    """
    Get the total number of trainable parameters in a model
    :param model:
    :return:
    """
    return sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )


def get_connections(model: torch.nn.Module):
    """
    Gets a list of the consecutive weight matrices in a given module
    :param model:
    :return:
    """
    children = list(model.children())
    return [(n,p.squeeze(),model) for n,p in model.named_parameters() if p.requires_grad and n not in ['bias', 'logsigma'] and len(p.shape) > 1 or n == 'mu'] if len(children) == 0 else [ci for c in children for ci in get_connections(c)]


def get_num_removed_params(model, vit=False):
    """
    Gets the number of parameters removed from a model based on which neurons are pruned
    :param model: A module with pruning layers
    :param vit: Whether or not the module is a Vision Transformer
    :return: The number of parameters which are removed from a model
    """
    connections = get_connections(model)
    vit_count = 0
    all_weights = []
    removed = 0
    for i, (n, p, m) in enumerate(connections):
        if n == 'mu':
            # Make sure the input is correct
            assert connections[i-1][1].shape[0] == connections[i][1].shape[0]
            # If the output matches then count it, otherwise don't
            prev_mask = m.mask.squeeze()
            # Remove mu and logsigma
            removed += 2 * ((prev_mask == 0).sum())
            for _ in connections[i-1][1].shape[1:]:
                prev_mask = prev_mask.unsqueeze(-1)
            connections[i-1][1].data *= prev_mask
            if connections[i+1][1].shape[1] == connections[i][1].shape[0]:
                next_mask = (m.mask).squeeze().unsqueeze(0)
                for _ in connections[i + 1][1].shape[2:]:
                    next_mask = next_mask.unsqueeze(-1)

                # Remove the output connections as well
                if vit and vit_count < 11:
                    connections[i + 1][1].data *= next_mask
                    # connections[i + 2][1].data *= next_mask
                    # connections[i + 3][1].data *= next_mask
                    vit_count += 1
                else:
                    connections[i+1][1].data *= next_mask
        else:
            all_weights.append(p)

    for w in all_weights:
        removed += (w == 0).sum()

    return removed


def get_true_parameter_compression_percent(model, vit=False):
    """
    Determine the number of trainable parameters pruned from a given model
    :param model: A module with pruning layers
    :param vit: Whether or not the module is a Vision Transformer
    :return: The number of percentage of trainable parameters that are removed from a model
    """
    n_removed = get_num_removed_params(model, vit)
    total_params = total_parameters(model)

    return (n_removed / total_params).item()