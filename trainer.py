from transformers import Trainer
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
        return (loss, outputs) if return_outputs else loss
