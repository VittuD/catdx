# trainer.py

from typing import Union
from src.trainers.TrainingArguments_projection import TrainingArguments_projection
from transformers import Trainer, TrainingArguments, PreTrainedModel
from src.utils.utils import compute_r2, compute_mae, compute_std, compute_mse
from scipy.stats import pearsonr
import torch
from torch import nn
from src.losses.contrastive import kernelized_supcon_loss

class LogTrainer(Trainer):
        
    def __init__(self,
                model: Union[PreTrainedModel, nn.Module] = None,
                args: TrainingArguments_projection = None,
                **kwargs):
        
        # print("LogTrainer init")
        # print(f"args: {args}")
        # print(f"kwargs: {kwargs}")

        # Get attributes from args
        self.training_mode = args.training_mode
        self.kernel_type = args.kernel_type
        self.contrastive_sigma = args.contrastive_sigma

        model.set_training_mode(self.training_mode)

        super().__init__(model=model, args=args, **kwargs)

        self.label_names+=['labels']
        self.epoch_wise_predictions = torch.tensor([])
        self.epoch_wise_labels = torch.tensor([])
        # if hasattr(self.model, 'set_training_mode'):
        #     self.model.set_training_mode(self.training_mode)

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

        # Extract features (e.g., 'projections') if available for contrastive loss
        features = outputs.get("projections")  # returns None if key is absent

        # Determine whether to use MSE loss
        use_mse = features is None or 'regression' in model.training_mode

        # Warn if features are missing
        if features is None:
            print("Warning: 'projections' not found. Defaulting to regression (MSE loss), which may be unintended.")

        if use_mse:
            loss = self.mse_loss(outputs, labels)
        else:
            loss = kernelized_supcon_loss(
                features=features.unsqueeze(1),  # Add extra dimension: [bsz, n_views, n_feats]
                labels=labels,
                kernel_type=self.kernel_type,    # Options: 'gaussian', 'rbf', or 'cauchy'
                temperature=0.07,
                sigma=self.contrastive_sigma,
                method='expw',                  # Options: 'expw', 'threshold', 'supcon', etc.
                contrast_mode='all',
                base_temperature=0.07,
                delta_reduction='sum'
            )

        # Log predictions and labels for r2 calculation
        predictions = outputs['logits'].squeeze()
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
        self.epoch_wise_predictions = torch.cat(
            (self.epoch_wise_predictions, predictions.detach().cpu())
        )
        self.epoch_wise_labels = torch.cat(
            (self.epoch_wise_labels, labels.detach().cpu())
        )

        return (loss, outputs) if return_outputs else loss
