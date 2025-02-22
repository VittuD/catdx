# contrastive.py

import math
import torch
from typing import Literal
from losses import KernelizedSupCon

def gaussian_kernel(labels, sigma: float = 1.0):
    """
    Compute a Gaussian kernel on the 1D labels. 
    labels: shape [batch_size, 1] or [batch_size].
    returns: [batch_size, batch_size] matrix of Gaussian weights.
    """
    labels = labels.float().view(-1, 1)  # ensure [bsz, 1]
    diff = labels - labels.T  # shape [bsz, bsz]
    # Gaussian kernel
    return torch.exp(- (diff ** 2) / (2.0 * (sigma ** 2))) / (math.sqrt(2.0 * math.pi) * sigma)

def rbk_kernel(labels, sigma: float = 1.0):
    """
    Compute a Radial Basis Kernel on the 1D labels.
    labels: shape [batch_size, 1] or [batch_size].
    returns: [batch_size, batch_size] matrix of RBF weights.
    """
    labels = labels.float().view(-1, 1)  # ensure [bsz, 1]
    diff = labels - labels.T             # shape [bsz, bsz]
    # RBF kernel equivalent to gamma = 1/(2*sigma^2)
    return torch.exp(- (diff ** 2) / (2 * sigma ** 2))

def cauchy_kernel(labels, sigma: float = 1.0):
    """
    Compute a Cauchy kernel on the 1D labels.
    labels: shape [batch_size, 1] or [batch_size].
    returns: [batch_size, batch_size] matrix of Cauchy weights.
    """
    labels = labels.float().view(-1, 1)  # ensure [bsz, 1]
    diff = labels - labels.T  # shape [bsz, bsz]
    # Cauchy kernel
    return 1.0 / (1.0 + (diff ** 2) / (sigma ** 2))

def kernelized_supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    kernel_type: Literal['gaussian', 'rbf', 'cauchy'] = 'gaussian',
    sigma: float = 1.0,
    method: str = 'expw',
    contrast_mode: str = 'all',
    base_temperature: float = 0.07,
    delta_reduction: str = 'sum',
):
    """
    A wrapper to instantiate and call KernelizedSupCon with the chosen kernel
    for method != 'supcon'. If method='supcon', no kernel is used.

    Args:
        features (torch.Tensor): shape [batch_size, n_views, n_features]
        labels (torch.Tensor): shape [batch_size]
        kernel_type (Literal): 'gaussian', 'rbf', or 'cauchy'
        sigma (float): standard deviation for the kernel
        method (str): 'supcon', 'threshold', 'expw', etc.
        contrast_mode (str): 'all' or 'one'
        base_temperature (float): base temperature (usually same as temperature)
        delta_reduction (str): 'mean' or 'sum', used in threshold method
    Returns:
        torch.Tensor: contrastive loss scalar
    """
    if method == 'supcon':
        # For plain supcon, kernel must be None
        kernel = None
    else:
        # Select kernel based on the provided kernel_type
        if kernel_type == 'gaussian':
            def kernel_fn(labels):
                return gaussian_kernel(labels, sigma=sigma)
        elif kernel_type == 'rbf':
            def kernel_fn(labels):
                return rbk_kernel(labels, sigma=sigma)
        elif kernel_type == 'cauchy':
            def kernel_fn(labels):
                return cauchy_kernel(labels, sigma=sigma)
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")
        kernel = kernel_fn

    # Create the loss function object
    loss_fn = KernelizedSupCon(
        method=method,
        temperature=base_temperature,
        contrast_mode=contrast_mode,
        base_temperature=base_temperature,
        kernel=kernel,
        delta_reduction=delta_reduction,
    )

    # Normalize the features for dot product
    normalized_features = torch.nn.functional.normalize(features, dim=-1, p=2)

    # Forward pass
    return loss_fn.forward(normalized_features, labels)
