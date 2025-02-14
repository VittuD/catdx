import datasets
import torch
from torcheval.metrics import R2Score
import numpy as np
from transformers import VivitImageProcessor

def get_image_processor(resize_to):
    return VivitImageProcessor(
        do_resize=True,
        size={'height': resize_to, 'width': resize_to},
        do_center_crop=False,
        do_normalize=True,
    )

# Dataset Utilities
def load_dataset(dataset_folder):
    dataset = datasets.load_dataset(dataset_folder)
    dataset = dataset.rename_column('video', 'pixel_values')
    dataset = dataset.rename_column('CO', 'labels')
    return dataset

def collate_fn(examples, image_processor):
    pixel_values = torch.stack([preprocess_example(example, image_processor).squeeze(0) for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def preprocess_example(example, image_processor, num_frames=32):
    video = example['pixel_values']
    frames = [frame.asnumpy() for frame in video]
    if len(frames) < num_frames:
        frames = frames * (num_frames // len(frames)) + frames[:num_frames % len(frames)]
    else:
        frames = frames[:num_frames]
    processed_video = image_processor(frames, return_tensors='pt')
    return processed_video['pixel_values']

# Metric Computation Utilities
def compute_mae(predictions, labels):
    return torch.mean(torch.abs(predictions - labels)).item()

# TODO what are we doing here...
def compute_std(predictions):
    return torch.std(predictions).item()

def compute_mse(predictions, labels):
    return torch.mean((predictions - labels) ** 2).item()

def compute_r2(predictions, labels):
    metric = R2Score()
    metric.to("cuda")
    if predictions.dim() == 2 and predictions.size(1) == 1:  # Shitty workaround
        predictions = predictions.squeeze(1)
    metric.update(predictions, labels)
    return metric.compute().item()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions) if isinstance(predictions, np.ndarray) else predictions
    # if predictions is a tuple take the first and convert to tensor
    predictions = torch.tensor(predictions[0]) if isinstance(predictions, tuple) else predictions
    labels = torch.tensor(labels) if isinstance(labels, np.ndarray) else labels
    
    # if labels is a tuple take the first and convert to tensor
    labels = torch.tensor(labels[0]) if isinstance(labels, tuple) else labels
    # Compute Pearson correlation using torch.corrcoef
    pearson_corr = torch.corrcoef(torch.stack((predictions.flatten(), labels.flatten())))[0, 1].item()

    return {
        "mae_e": compute_mae(predictions, labels),
        "std_e": compute_std(predictions),
        "mse_e": compute_mse(predictions, labels),
        "r2_e": compute_r2(predictions, labels),
        "pearson_e": float(pearson_corr),
    }

# Overwrite 'eval' to 'val' in logs (still fails but it's no priority)
# def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="val"):
#        return super().evaluate(self, eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
