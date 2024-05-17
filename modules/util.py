import torch
from torch.special import erf
from torch.special import ndtri
import numpy as np


"""
Math for log-uniform dropout layer. Much of this is pulled from Neklyudov et al. 2017, and can be found 
here: https://github.com/necludov/group-sparsity-sbp/blob/master/nets/layers.py
"""
def getcoord(idx, dims):
    coords = []
    for dim in dims[::-1]:
        coords.append(idx % dim)
        idx = int(idx / dim)
    return coords[::-1]


def phi(x):
    return torch.exp(-0.5 * x**2) / np.sqrt(2*torch.pi)


def Phi(x):
    return 0.5 * (1 + erf(x / np.sqrt(2.)))


def erfinv(x):
    return ndtri((x + 1.) / 2.0) / np.sqrt(2.)


def Phi_inv(x):
    return np.sqrt(2.0)*torch.special.erfinv(2.*x-1)#erfinv(2.0*x-1)


def erfcx(x):
    """M. M. Shepherd and J. G. Laframboise,
       MATHEMATICS OF COMPUTATION 36, 249 (1981)
    """
    K = 3.75
    y = (torch.abs(x)-K) / (torch.abs(x)+K)
    y2 = 2.0*y
    (d, dd) = (-0.4e-20, 0.0)
    (d, dd) = (y2 * d - dd + 0.3e-20, d)
    (d, dd) = (y2 * d - dd + 0.97e-19, d)
    (d, dd) = (y2 * d - dd + 0.27e-19, d)
    (d, dd) = (y2 * d - dd + -0.2187e-17, d)
    (d, dd) = (y2 * d - dd + -0.2237e-17, d)
    (d, dd) = (y2 * d - dd + 0.50681e-16, d)
    (d, dd) = (y2 * d - dd + 0.74182e-16, d)
    (d, dd) = (y2 * d - dd + -0.1250795e-14, d)
    (d, dd) = (y2 * d - dd + -0.1864563e-14, d)
    (d, dd) = (y2 * d - dd + 0.33478119e-13, d)
    (d, dd) = (y2 * d - dd + 0.32525481e-13, d)
    (d, dd) = (y2 * d - dd + -0.965469675e-12, d)
    (d, dd) = (y2 * d - dd + 0.194558685e-12, d)
    (d, dd) = (y2 * d - dd + 0.28687950109e-10, d)
    (d, dd) = (y2 * d - dd + -0.63180883409e-10, d)
    (d, dd) = (y2 * d - dd + -0.775440020883e-09, d)
    (d, dd) = (y2 * d - dd + 0.4521959811218e-08, d)
    (d, dd) = (y2 * d - dd + 0.10764999465671e-07, d)
    (d, dd) = (y2 * d - dd + -0.218864010492344e-06, d)
    (d, dd) = (y2 * d - dd + 0.774038306619849e-06, d)
    (d, dd) = (y2 * d - dd + 0.4139027986073010e-05, d)
    (d, dd) = (y2 * d - dd + -0.69169733025012064e-04, d)
    (d, dd) = (y2 * d - dd + 0.490775836525808632e-03, d)
    (d, dd) = (y2 * d - dd + -0.2413163540417608191e-02, d)
    (d, dd) = (y2 * d - dd + 0.9074997670705265094e-02, d)
    (d, dd) = (y2 * d - dd + -0.26658668435305752277e-01, d)
    (d, dd) = (y2 * d - dd + 0.59209939998191890498e-01, d)
    (d, dd) = (y2 * d - dd + -0.84249133366517915584e-01, d)
    (d, dd) = (y2 * d - dd + -0.4590054580646477331e-02, d)
    d = y * d - dd + 0.1177578934567401754080e+01
    result = d/(1.0+2.0*torch.abs(x))
    result = torch.where(torch.isnan(result), torch.ones_like(result), result)
    result = torch.where(torch.isinf(result), torch.ones_like(result), result)

    negative_mask = (x < 0.0).type(torch.float32)
    positive_mask = (x >= 0.0).type(torch.float32)
    negative_result = 2.0*torch.exp(x**2)-result
    negative_result = torch.where(torch.isnan(negative_result), torch.ones_like(negative_result), negative_result)
    negative_result = torch.where(torch.isinf(negative_result), torch.ones_like(negative_result), negative_result)
    result = negative_mask * negative_result + positive_mask * result
    return result


def LogU_pdf(x, a, b):
    return 1 / (torch.exp(x) * (b - a))


def LogU_cdf(x1, x2, a, b):
    return (x2 - x1) / (b - a)


def LogN_cdf(x1, x2, mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    xi1 = (x1 - mu) / sigma
    xi2 = (x2 - mu) / sigma

    cdf_alpha = Phi(alpha)
    cdf_beta = Phi(beta)
    Z = cdf_beta - cdf_alpha

    return (Phi(xi2) - Phi(xi1)) / Z


def LogN_pdf(x, mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    cdf_alpha = Phi(alpha)
    cdf_beta = Phi(beta)
    Z = cdf_beta - cdf_alpha

    return 1 / (torch.exp(x) * torch.sqrt(2 * torch.pi * sigma**2) * Z) * torch.exp(-0.5 * (x - mu)**2 / sigma**2)


def reparameterize(mu, sigma, a, b, y, eps=1e-5):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    cdf_alpha = Phi(alpha)
    cdf_beta = Phi(beta)
    Z = cdf_beta - cdf_alpha
    gamma = torch.clamp(cdf_alpha + y * Z, min=eps, max=1 - eps)

    return torch.clamp(mu + sigma * Phi_inv(gamma), min=a, max=b).exp()


def kldiv(mu, sigma, a, b, eps=1e-5):
    # Calculate \alpha, \beta
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    # Get CDFs
    cdf_alpha = Phi(alpha)
    cdf_beta = Phi(beta)
    Z = cdf_beta - cdf_alpha
    # Get PDFs
    pdf_alpha = phi(alpha)
    pdf_beta = phi(beta)

    # Log terms
    term1 = -torch.log(sigma) - torch.log(Z) - np.log(2.0 * torch.pi * torch.e) / 2.0 + np.log(b - a)

    # Last term
    term2 = (alpha * pdf_alpha - beta * pdf_beta) / (2 * Z)

    return term1 - term2


def SNR(mu, sigma, a, b, eps=1e-5):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Z = Phi(beta) - Phi(alpha)
    sq2 = np.sqrt(2)
    
    ratio = erfcx((sigma - beta) / sq2) * torch.exp((b - mu) - beta ** 2 / 2.0)
    ratio = ratio - erfcx((sigma - alpha) / sq2) * torch.exp((a - mu) - alpha ** 2 / 2.0)
    denominator = 2 * Z * erfcx((2.0 * sigma - beta) / sq2) * torch.exp(2.0 * (b - mu) - beta ** 2 / 2.0)
    denominator = denominator - 2 * Z * erfcx((2.0 * sigma - alpha) / sq2) * torch.exp(
        2.0 * (a - mu) - alpha ** 2 / 2.0)
    denominator = denominator - ratio ** 2
    ratio = ratio / torch.sqrt(denominator)
    return ratio


def Etheta(mu, sigma, a, b, eps=1e-5):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Z = Phi(beta) - Phi(alpha)
    sq2 = np.sqrt(2)

    mean = erfcx((sigma - beta) / sq2) * torch.exp(b - beta * beta / 2)
    mean = mean - erfcx((sigma - alpha) / sq2) * torch.exp(a - alpha * alpha / 2)
    mean = mean / (2 * Z)
    return mean


def Var(mu, sigma, a, b, eps=1e-5):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Z = Phi(beta) - Phi(alpha)

    term1 = torch.exp(2*mu + sigma**2)
    term2 = torch.exp(sigma**2)*(Phi(2*sigma - alpha) - Phi(2*sigma - beta))
    term3 = (1 / Z) * (Phi(sigma - alpha) - Phi(sigma - beta))**2

    return term1 * (term2 - term3)

