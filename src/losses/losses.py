"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 21/09/23
"""

from cmath import isinf
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

import wandb
import numpy as np
import matplotlib.pyplot as plt

# DEBUG ONLY
def log_data_as_table_and_heatmap(data, key_table="logged_table", key_heatmap="matrix_heatmap", columns=None):
    """
    Creates a wandb.Table from the provided data and logs it.
    Also creates a heatmap from the data using matplotlib, places the x-axis legend on top,
    and logs the figure.

    Args:
        data (list or numpy array): 2D data (e.g., a matrix) to create the table and heatmap from.
        key_table (str, optional): The key under which to log the table. Defaults to "logged_table".
        key_heatmap (str, optional): The key under which to log the heatmap. Defaults to "matrix_heatmap".
        columns (list, optional): List of column names. If not provided, columns will be auto-generated.
    """
    # Convert numpy array to list if needed.
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    else:
        data_list = data

    # Auto-generate column names if not provided.
    if columns is None:
        if data_list and isinstance(data_list[0], list):
            num_cols = len(data_list[0])
        else:
            # Handle 1D data by converting it to a column vector.
            data_list = [[item] for item in data_list]
            num_cols = 1
        columns = [f"col_{i}" for i in range(num_cols)]
    
    # Create and log the table.
    # table = wandb.Table(data=data_list, columns=columns)
    # wandb.log({key_table: table})
    
    # Create a heatmap using matplotlib.
    fig, ax = plt.subplots()
    # Convert the data to a numpy array for imshow.
    heatmap_data = np.array(data_list)
    # Fix the range of the plot from 0 to -1
    cax = ax.imshow(heatmap_data, cmap='viridis', vmin=-1, vmax=0)
    fig.colorbar(cax)
    ax.set_title("Heatmap")
    
    # Move x-axis ticks and labels to the top.
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Log the heatmap figure.
    wandb.log({key_heatmap: fig})
    
    # Close the figure to free memory.
    plt.close(fig)


class KernelizedSupCon(nn.Module):
    """Supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Based on: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, method: str, temperature: float=0.07, contrast_mode: str='all',
                 base_temperature: float=0.07, kernel: callable=None, delta_reduction: str='sum'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.method = method
        self.kernel = kernel
        self.delta_reduction = delta_reduction

        if kernel is not None and method == 'supcon':
            raise ValueError('Kernel must be none if method=supcon')
        
        if kernel is None and method != 'supcon':
            raise ValueError('Kernel must not be none if method != supcon')

        if delta_reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction {delta_reduction}")

    def __repr__(self):
        return f'{self.__class__.__name__} ' \
               f'(t={self.temperature}, ' \
               f'method={self.method}, ' \
               f'kernel={self.kernel is not None}, ' \
               f'delta_reduction={self.delta_reduction})'

    def forward(self, features, labels=None):
        """Compute loss for model. If `labels` is None, 
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, n_features]. 
                input has to be rearranged to [bsz, n_views, n_features] and labels [bsz],
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) != 3:
            raise ValueError('`features` needs to be [bsz, n_views, n_feats],'
                             '3 dimensions are required')

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is None:
            mask = torch.eye(batch_size, device=device)
        
        else:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            if self.kernel is None:
                mask = torch.eq(labels, labels.T)
            else:
                mask = self.kernel(labels)
            
        view_count = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            features = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            features = features
            anchor_count = view_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Tile mask
        mask = mask.repeat(anchor_count, view_count)

        # Inverse of torch-eye to remove self-contrast (diagonal)
        inv_diagonal = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size*n_views, device=device).view(-1, 1),
            0
        )

        # compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Log logits matrix on wandb on current step
        if wandb.run is not None:
            log_data_as_table_and_heatmap(logits.cpu().detach().numpy(), key_table="logits_matrix"
            "step_" + str(wandb.run.step))

        alignment = logits 

        # base case is:
        # - supcon if kernel = none 
        # - y-aware is kernel != none
        uniformity = torch.exp(logits) * inv_diagonal 

        if self.method == 'threshold':
            repeated = mask.unsqueeze(-1).repeat(1, 1, mask.shape[0]) # repeat kernel mask

            delta = (mask[:, None].T - repeated.T).transpose(1, 2) # compute the difference w_k - w_j for every k,j
            delta = (delta > 0.).float()

            # for each z_i, repel only samples j s.t. K(z_i, z_j) < K(z_i, z_k)
            uniformity = uniformity.unsqueeze(-1).repeat(1, 1, mask.shape[0])

            if self.delta_reduction == 'mean':
                uniformity = (uniformity * delta).mean(-1)
            else:
                uniformity = (uniformity * delta).sum(-1)
    
        elif self.method == 'expw':
            # exp weight e^(s_j(1-w_j))
            uniformity = torch.exp(logits * (1 - mask)) * inv_diagonal

        uniformity = torch.log(uniformity.sum(1, keepdim=True))

        # Log uniformity matrix on wandb on current step
        # if wandb.run is not None:
        #     log_data_as_table(uniformity.cpu().detach().numpy(), key="uniformity_matrix"
        #     "step_" + str(wandb.run.step))

        # positive mask contains the anchor-positive pairs
        # excluding <self,self> on the diagonal
        positive_mask = mask * inv_diagonal

        log_prob = alignment - uniformity # log(alignment/uniformity) = log(alignment) - log(uniformity)
        log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(1) # compute mean of log-likelihood over positive
 
        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob
        return loss.mean()


if __name__ == '__main__':
    k_supcon = KernelizedSupCon(1.0)

    x = torch.nn.functional.normalize(torch.randn((256, 2, 64)), dim=1)
    labels = torch.randint(0, 4, (256,))

    l = k_supcon(x, labels)
    print(l)
