# contrastive.py

import math
import torch
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

def kernelized_supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    sigma: float = 1.0,
    method: str = 'expw',
    contrast_mode: str = 'all',
    base_temperature: float = 0.07,
    delta_reduction: str = 'sum',
):
    """
    A wrapper to instantiate and call KernelizedSupCon with a Gaussian kernel
    for method != 'supcon'. If method='supcon', no kernel is used.

    Args:
        features (torch.Tensor): shape [batch_size, n_views, n_features]
        labels (torch.Tensor): shape [batch_size]
        temperature (float): temperature scaling
        sigma (float): standard deviation for the Gaussian kernel
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
        # For other methods, define the kernel function
        def kernel_fn(labels):
            return gaussian_kernel(labels, sigma=sigma)
        kernel = kernel_fn

    # Create the loss function object
    loss_fn = KernelizedSupCon(
        method=method,
        temperature=temperature,
        contrast_mode=contrast_mode,
        base_temperature=base_temperature,
        kernel=kernel,
        delta_reduction=delta_reduction,
    )

    # Forward pass
    return loss_fn.forward(features, labels)
