import torch
from torcheval.metrics import R2Score
import numpy as np
from transformers import VivitImageProcessor, TrainerCallback
import torchvision.transforms.functional as F
from PIL import Image
import cv2
from decord import VideoReader
import os
import pandas as pd
from datasets import Dataset, Features, Value, Video, DatasetDict
import datasets

VivitImageProcessor()

def get_image_processor(resize_to, num_channels):
    # Generate a list with num_channels times elements 0.5
    image_mean = [0.5] * num_channels
    image_std = [0.5] * num_channels

    return VivitImageProcessor(
        do_resize=True,
        size={'height': resize_to, 'width': resize_to},
        do_center_crop=False,
        do_normalize=True,
        image_mean=image_mean,
        image_std=image_std,
    )

# Dataset Utilities
def create_datasets(csv_file, video_folder):
        df = pd.read_csv(csv_file)
        df["file_name"] = df["file_name"].apply(lambda x: os.path.abspath(os.path.join(video_folder, x)))

        features = Features({
            "file_name": Video(),
            "CO": Value("float"),
            "partition": Value("string"),
        })

        dataset = Dataset.from_pandas(df, features=features)
        train_dataset = dataset.filter(lambda x: x["partition"] == "train")
        val_dataset = dataset.filter(lambda x: x["partition"] == "val")
        test_dataset = dataset.filter(lambda x: x["partition"] == "test")
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })

def load_dataset(dataset_folder, augmentation=None):
    # Augmentation not implemented yet

    # Old way, works with apical4_none dir structure
    # dataset = datasets.load_dataset(dataset_folder)
    # dataset = dataset.rename_column('video', 'pixel_values')

    # New way, works with preprocessed/mp4/apical4 dir structure
    dataset = create_datasets(os.path.join(dataset_folder, "all_files_with_partition.csv"), dataset_folder)
    dataset = dataset.rename_column('file_name', 'pixel_values')

    dataset = dataset.rename_column('CO', 'labels')
    return dataset

def collate_fn(examples, image_processor, num_channels):
    pixel_values = torch.stack([preprocess_example(example, image_processor, num_channels).squeeze(0) for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def convert_to_grayscale(video_frames):
    # Convert each frame from RGB to grayscale (luminance only)
    return [Image.fromarray(frame).convert("L") for frame in video_frames]

def preprocess_example(example, image_processor, num_channels=1, num_frames=32):
    video = example['pixel_values']
    original_frames = [frame.asnumpy() for frame in video]
    if num_channels == 1:
        original_frames = convert_to_grayscale(original_frames)
        original_frames = [np.array(frame)[..., None] for frame in original_frames]
    frames = original_frames
    # Save the video back as mp4 with cv2 for manual sanity check
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('original.mp4', fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))
    # for frame in frames:
    #     out.write(frame)
    # out.release()
    
    if len(frames) < num_frames:
        frames = frames * (num_frames // len(frames)) + frames[:num_frames % len(frames)]
    else:
        frames = frames[:num_frames]
    
    processed_video = image_processor(frames, return_tensors='pt')
    return processed_video['pixel_values']

# Metric Computation Utilities
def compute_mae(predictions, labels):
    return torch.mean(torch.abs(predictions - labels)).item()

def compute_std(predictions):
    return torch.std(predictions).item()

def compute_mse(predictions, labels):
    return torch.mean((predictions - labels) ** 2).item()

def compute_r2(predictions, labels):
    metric = R2Score()
    metric.to("cuda")
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
