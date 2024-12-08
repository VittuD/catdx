import json
import datasets
import torch
import numpy as np
from transformers import VivitImageProcessor

# Load configuration from a JSON file
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

# Dataset Utilities
def load_dataset(dataset_folder):
    dataset = datasets.load_dataset(dataset_folder)
    dataset = dataset.rename_column('video', 'pixel_values')
    dataset = dataset.rename_column('CO', 'labels')
    return dataset

def preprocess_example(example, image_processor):
    video = example['pixel_values']
    frames = [frame.asnumpy() for frame in video]
    processed_video = image_processor(frames, return_tensors='pt')
    return processed_video['pixel_values']

def get_image_processor(resize_to):
    return VivitImageProcessor(
        do_resize=True,
        size={'height': resize_to, 'width': resize_to},
        do_center_crop=False,
        do_normalize=True,
    )

def collate_fn(examples, image_processor):
    pixel_values = torch.stack([preprocess_example(example, image_processor).squeeze(0) for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# Metric Computation Utilities
def compute_mae(predictions, labels):
    return torch.mean(torch.abs(predictions - labels)).item()

def compute_std(predictions, labels):
    return torch.std(predictions - labels).item()

def compute_mse(predictions, labels):
    return torch.mean((predictions - labels) ** 2).item()

def compute_pearson_r2(predictions, labels):
    mean_predictions = torch.mean(predictions)
    mean_labels = torch.mean(labels)
    covariance = torch.mean((predictions - mean_predictions) * (labels - mean_labels))
    variance_predictions = torch.mean((predictions - mean_predictions) ** 2)
    variance_labels = torch.mean((labels - mean_labels) ** 2)
    if variance_predictions == 0 or variance_labels == 0:
        return float('nan')
    return (covariance / (torch.sqrt(variance_predictions * variance_labels))) ** 2

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions) if isinstance(predictions, np.ndarray) else predictions
    labels = torch.tensor(labels) if isinstance(labels, np.ndarray) else labels
    return {
        "mae": compute_mae(predictions, labels),
        "std": compute_std(predictions, labels),
        "mse": compute_mse(predictions, labels),
        "pearson_r2": compute_pearson_r2(predictions, labels),
    }
