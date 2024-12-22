import torch
import torch.nn.functional as F

# Gaussian kernel function
def gaussian_kernel(x, sigma):
    """
    Compute the Gaussian kernel matrix.
    Args:
        x (torch.Tensor): Input tensor with shape [N, d].
        sigma (float): The standard deviation for the Gaussian kernel.
    Returns:
        torch.Tensor: Kernel matrix with shape [N, N].
    """
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    sq_distances = torch.sum(diff ** 2, dim=-1)
    return torch.exp(-sq_distances / (2 * (sigma ** 2)))

# Gaussian contrastive loss function
def gaussian_contrastive_loss(features, labels, temperature=0.07, sigma=1.0, method='exp'):
    """
    Compute contrastive loss with a Gaussian kernel.

    Args:
        features (torch.Tensor): Feature embeddings with shape [batch_size, feature_dim].
        labels (torch.Tensor): Labels with shape [batch_size].
        temperature (float): Temperature scaling for similarity.
        sigma (float): Standard deviation for Gaussian kernel.
        method (str): Loss method ('y-aware' or 'exp').
    Returns:
        torch.Tensor: The contrastive loss.
    """
    device = features.device
    batch_size = features.size(0)

    # Normalize features
    features = F.normalize(features, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Gaussian kernel applied to labels
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    kernel_matrix = gaussian_kernel(label_diff, sigma)

    # Positive mask
    positive_mask = kernel_matrix.clone()

    # Apply y-aware or Lexp formulations
    if method == 'y-aware':
        # y-aware: Normalize kernel weights
        kernel_sum = kernel_matrix.sum(dim=1, keepdim=True)
        normalized_kernel = kernel_matrix / kernel_sum

        # Log-sum-exp formulation
        log_prob = similarity_matrix - torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
        log_prob = normalized_kernel * log_prob

    elif method == 'exp':
        # Lexp: Adjust weights inversely proportional to kernel distance
        adjusted_weights = 1 - kernel_matrix
        log_prob = similarity_matrix - torch.logsumexp(similarity_matrix * adjusted_weights, dim=1, keepdim=True)
        log_prob = kernel_matrix * log_prob

    else:
        raise ValueError(f"Invalid method: {method}. Choose 'y-aware' or 'exp'.")

    # Avoid self-contrast
    mask = torch.eye(batch_size, device=device)
    positive_mask -= mask

    # Compute mean loss
    loss = -torch.sum(log_prob) / torch.sum(positive_mask)

    return loss
