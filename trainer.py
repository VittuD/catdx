from transformers import Trainer
from utils import compute_pearson_r2, compute_mae, compute_std
import torch

class LogTrainer(Trainer):
    def log(self, logs, start_time='NaN'):
        logs["learning_rate"] = self._get_learning_rate()
        logs["step"] = self.state.global_step
        super().log(logs)

    def compute_loss(self, model, inputs, num_items_in_batch=1, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        predictions = outputs.logits.squeeze()
        loss = torch.nn.functional.mse_loss(predictions, labels)
        r2 = compute_pearson_r2(predictions, labels)
        mae = compute_mae(predictions, labels)
        std = compute_std(predictions, labels)
        return (loss, outputs) if return_outputs else loss
