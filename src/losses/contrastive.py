# contrastive.py

import math
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from typing import Literal
from src.losses.losses import KernelizedSupCon

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
    temperature: float = 0.07,
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
        elif kernel_type == 'none':
            kernel_fn = None
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")
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

    # Normalize the features for dot product
    normalized_features = F.normalize(features, dim=-1, p=2)

    # Forward pass
    return loss_fn.forward(normalized_features, labels)


# Plotting helper functions

def plot_and_save_kernel_heatmap(kernel_matrix: torch.Tensor, labels: torch.Tensor, 
                                 title: str, filename: str):
    """
    Plots a heatmap with Plotly, labeling both axes by the actual label values,
    and saves to a file (PNG, JPEG, PDF, SVG, etc.) without displaying.
    Requires kaleido or orca for static image export.
    """
    # Convert torch Tensors to numpy
    z_vals = kernel_matrix.detach().cpu().numpy()
    # Use labels themselves for axes
    axis_vals = labels.detach().cpu().tolist()

    fig = go.Figure(
        data=go.Heatmap(
            x=axis_vals,
            y=axis_vals,
            z=z_vals,
            colorscale='Viridis'
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='Labels',
        yaxis_title='Labels',
        title_x=0.5,
    )

    # Save to file (e.g. PNG). For HTML export, you can use fig.write_html().
    fig.write_image(filename)

# Testing main
def main():
    # Define some dummy data
    features = torch.randn(5, 5, 1)
    labels = torch.rand(5)
    sigma = 0.1
    print("Features:")
    print(features)
    print("Labels:")
    print(labels)

    # Compute kernel matrices
    gaussian_kernel_results = gaussian_kernel(labels, sigma=sigma)
    rbf_kernel_results = rbk_kernel(labels, sigma=sigma)
    cauchy_kernel_results = cauchy_kernel(labels, sigma=sigma)

    # Print and save each kernel matrix
    print("Gaussian Kernel:")
    print(gaussian_kernel_results)
    plot_and_save_kernel_heatmap(gaussian_kernel_results, labels, 
                                 "Gaussian Kernel Heatmap", "gaussian_kernel.png")

    print("RBF Kernel:")
    print(rbf_kernel_results)
    plot_and_save_kernel_heatmap(rbf_kernel_results, labels, 
                                 "RBF Kernel Heatmap", "rbf_kernel.png")

    print("Cauchy Kernel:")
    print(cauchy_kernel_results)
    plot_and_save_kernel_heatmap(cauchy_kernel_results, labels, 
                                 "Cauchy Kernel Heatmap", "cauchy_kernel.png")

    # Test the kernelized_supcon_loss function
    print("Testing KernelizedSupCon with Gaussian kernel:")
    loss_gaussian = kernelized_supcon_loss(features, labels, kernel_type='gaussian', sigma=sigma)
    print(loss_gaussian)

    print("Testing KernelizedSupCon with RBF kernel:")
    loss_rbf = kernelized_supcon_loss(features, labels, kernel_type='rbf', sigma=sigma)
    print(loss_rbf)

    print("Testing KernelizedSupCon with Cauchy kernel:")
    loss_cauchy = kernelized_supcon_loss(features, labels, kernel_type='cauchy', sigma=sigma)
    print(loss_cauchy)

if __name__ == "__main__":
    main()
