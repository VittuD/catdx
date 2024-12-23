from transformers import Trainer
from utils import compute_r2, compute_mae, compute_std, compute_mse
from scipy.stats import pearsonr
import torch
from contrastive import gaussian_contrastive_loss

class LogTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_wise_predictions = torch.tensor([])
        self.epoch_wise_labels = torch.tensor([])

    def log(self, logs, start_time='NaN'):
        logs["learning_rate"] = self._get_learning_rate()
        logs["step"] = self.state.global_step
        # Add train/r2 data leveraging the batch_wise_predictions and batch_wise_labels
        if self.state.is_local_process_zero:
            if self.epoch_wise_predictions.numel() > 0 and self.epoch_wise_labels.numel() > 0:
                logs["r2"] = compute_r2(self.epoch_wise_predictions, self.epoch_wise_labels)
                logs["pearson"] = float(pearsonr(self.epoch_wise_predictions, self.epoch_wise_labels)[0])
                logs["mae"] = compute_mae(self.epoch_wise_predictions, self.epoch_wise_labels)
                logs["std"] = compute_std(self.epoch_wise_predictions, self.epoch_wise_labels)
                logs["mse"] = compute_mse(self.epoch_wise_predictions, self.epoch_wise_labels)
                self.epoch_wise_predictions = torch.tensor([])
                self.epoch_wise_labels = torch.tensor([])
        super().log(logs)

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Use projection head's output if available
        # TODO modify this accordingly to the model's forward method
        if hasattr(model, "projection_head") and model.projection_head is not None:
            features = model.projection_head(outputs.last_hidden_state.mean(dim=1))
        else:
            features = outputs.last_hidden_state.mean(dim=1)  # Default to mean pooling

        # Compute Gaussian contrastive loss
        loss = gaussian_contrastive_loss(features, labels, temperature=0.1, sigma=2.0, method='exp')

        # Log predictions and labels for r2 calculation
        predictions = (lambda x: x.unsqueeze(0) if x.dim() == 0 else x)(outputs.logits.squeeze())
        
        # Handle non-1 num_items_in_batch
        if num_items_in_batch > 1:
            predictions = predictions.view(-1, num_items_in_batch).mean(dim=1)
            labels = labels.view(-1, num_items_in_batch).mean(dim=1)
        
        self.epoch_wise_predictions = torch.cat((self.epoch_wise_predictions, predictions.detach().cpu()))
        self.epoch_wise_labels = torch.cat((self.epoch_wise_labels, labels.detach().cpu()))

        return (loss, outputs) if return_outputs else loss

    # Overriding the evaluate method to rename the metric to val instead of eval
    # def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="val"):
    #     super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    # TODO This uploads "train/val_..." metrics to wandb, but we want to upload "val/..." metrics
    # Maybe tied to how wandb interprets the metric_key_prefix
