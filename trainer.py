# trainer.py

from transformers import Trainer
from utils import compute_r2, compute_mae, compute_std, compute_mse
from scipy.stats import pearsonr
import torch

# Import the new wrapper function
from contrastive import kernelized_supcon_loss

class LogTrainer(Trainer):
    def __init__(self, training_mode='regression', *args, **kwargs):
        # Pop kernel_type from kwargs
        kernel_type = kwargs.pop('kernel_type', 'gaussian')
        super().__init__(*args, **kwargs)
        self.label_names+=['labels']
        self.training_mode = training_mode
        self.epoch_wise_predictions = torch.tensor([])
        self.epoch_wise_labels = torch.tensor([])
        self.kernel_type = kernel_type
        
    def log(self, logs, start_time='NaN'):
        logs["learning_rate"] = self._get_learning_rate()
        logs["step"] = self.state.global_step
        # Add train/r2 data leveraging the batch_wise_predictions and batch_wise_labels
        if self.state.is_local_process_zero:
            if self.epoch_wise_predictions.numel() > 0 and self.epoch_wise_labels.numel() > 0:
                # If logs has a key with 'eval' in it, set prefix to "eval_"
                prefix = "eval_" if any('eval' in key for key in logs.keys()) else ""
                logs[f"{prefix}r2"] = compute_r2(self.epoch_wise_predictions, self.epoch_wise_labels)
                logs[f"{prefix}pearson"] = float(pearsonr(self.epoch_wise_predictions, self.epoch_wise_labels)[0])
                logs[f"{prefix}mae"] = compute_mae(self.epoch_wise_predictions, self.epoch_wise_labels)
                logs[f"{prefix}std"] = compute_std(self.epoch_wise_predictions)
                logs[f"{prefix}mse"] = compute_mse(self.epoch_wise_predictions, self.epoch_wise_labels)
                self.epoch_wise_predictions = torch.tensor([])
                self.epoch_wise_labels = torch.tensor([])
        super().log(logs)

    def mse_loss(self, outputs, labels):
        predictions = (lambda x: x.unsqueeze(0) if x.dim() == 0 else x)(outputs['logits'].squeeze())
        loss = torch.nn.functional.mse_loss(predictions, labels)
        self.epoch_wise_predictions = torch.cat((self.epoch_wise_predictions, predictions.detach().cpu()))
        self.epoch_wise_labels = torch.cat((self.epoch_wise_labels, labels.detach().cpu()))
        return loss

    def compute_loss(self, model, inputs, num_items_in_batch=1, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Extract features from the model's output
        # Suppose 'outputs' has 'projections' if we want contrastive
        features = outputs['projections'] if 'projections' in outputs else None

        # If features is None, use MSE loss; else use kernelized_supcon_loss
        if self.training_mode == 'regression' or features is None:
            loss = self.mse_loss(outputs, labels)
        else:
            # Example usage: method='expw' with a bigger sigma
            loss = kernelized_supcon_loss(
                features=features.unsqueeze(1),  # Add an extra dimension [bsz, n_views, n_feats]
                labels=labels,
                kernel_type=self.kernel_type,  # 'gaussian', 'rbf', or 'cauchy'
                temperature=0.07, 
                sigma=1.0, 
                method='expw',    # or 'threshold' or 'supcon' ...
                contrast_mode='all',
                base_temperature=0.07,
                delta_reduction='sum'
            )
            
        # Log predictions and labels for r2 calculation
        predictions = (lambda x: x.unsqueeze(0) if x.dim() == 0 else x)(outputs['logits'].squeeze())
        self.epoch_wise_predictions = torch.cat((self.epoch_wise_predictions, predictions.detach().cpu()))
        self.epoch_wise_labels = torch.cat((self.epoch_wise_labels, labels.detach().cpu()))

        return (loss, outputs) if return_outputs else loss
