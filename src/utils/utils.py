import torch
from torcheval.metrics import R2Score
import numpy as np
from transformers import VivitImageProcessor, TrainerCallback
import torchvision.transforms.functional as F
from PIL import Image
import cv2
import os
import pandas as pd
from pathlib import Path
from datasets import Dataset, Features, Value, Video, Array4D, DatasetDict


def safe_cat(existing, new):
    # Return early if one of the inputs is None or empty
    if existing is None or (isinstance(existing, (list, tuple)) and not existing):
        return new
    if new is None or (isinstance(new, (list, tuple)) and not new):
        return existing

    # Ensure tensors are on CPU
    if isinstance(existing, torch.Tensor):
        existing = existing.cpu()
    if isinstance(new, torch.Tensor):
        new = new.cpu()

    # Convert lists/tuples to tensors
    if isinstance(existing, (list, tuple)):
        existing = torch.cat(existing, dim=0)
    if isinstance(new, (list, tuple)):
        new = torch.cat(new, dim=0)

    return torch.cat([existing, new], dim=0)


def get_image_processor(resize_to, num_channels):
    # Generate a list with num_channels times elements (Imagenet standard normalization)
    image_mean = [0.5] * num_channels
    image_std = [0.5] * num_channels

    return VivitImageProcessor(
        do_resize=True,
        size={'height': resize_to, 'width': resize_to},
        do_center_crop=False,
        do_normalize=False,
        do_rescale=False,
        offset=False,
        image_mean=image_mean,
        image_std=image_std,
    )

## Dataset Utilities
def create_datasets(csv_file, video_folder):
        df = pd.read_csv(csv_file)
        df["file_name"] = df["file_name"].apply(lambda x: os.path.abspath(os.path.join(video_folder, x)))

        features = Features({
            "file_name": Video("string"),
            "CO": Value("float"),
            "partition": Value("string"),
        })

        dataset = Dataset.from_pandas(df, features=features)
        train_dataset = dataset.filter(lambda x: x["partition"] == "train")
        val_dataset = dataset.filter(lambda x: x["partition"] == "validation")
        test_dataset = dataset.filter(lambda x: x["partition"] == "test")
        
        
        datadict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })

        # datadict.save_to_disk("saved_hf_dataset", max_shard_size="1GB")
        
        return datadict

def load_dataset(dataset_folder, augmentation=None):
    # Augmentation not implemented yet

    # Old way, works with apical4_none dir structure
    # dataset = datasets.load_dataset(dataset_folder)
    # dataset = dataset.rename_column('video', 'pixel_values')

    # New way, works with preprocessed/mp4/apical4 dir structure
    dataset = create_datasets(os.path.join(dataset_folder, "codebook_vivit.csv"), dataset_folder)
    dataset = dataset.rename_column('file_name', 'pixel_values')

    dataset = dataset.rename_column('CO', 'labels')
    return dataset


## Video/Batch Processing Utilities
def collate_fn(examples, image_processor, num_channels):
    pixel_values = torch.stack([
        preprocess_example(example, image_processor, num_channels).squeeze(0)
        for example in examples
    ])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def preprocess_example(example, image_processor, num_channels=1, num_frames=32):
    # NEW Huggingface release: video is a torchvision VideoReader object
    video = example['pixel_values']
    
    # Convert video frames to NumPy arrays using the as_numpy helper
    original_frames = as_numpy(video)
    
    if num_channels == 1:
        original_frames = convert_to_grayscale(original_frames)
    
    frames = original_frames
    
    # Adjust the number of frames to match num_frames (pad or truncate as needed)
    frames = adjust_frames(frames, num_frames)

    # Normalize from [0, 255] to [0, 1]
    frames = [frame / 255.0 for frame in frames]
    
    processed_video = image_processor(frames, return_tensors='pt')
    return processed_video['pixel_values']

def as_numpy(video):
    frames = []
    for frame in video:
        tensor_frame = frame['data']
        if tensor_frame.device.type != "cpu":
            tensor_frame = tensor_frame.cpu()
        # Convert tensor to NumPy array immediately
        frames.append(tensor_frame.numpy())
    return frames

def convert_to_grayscale(video_frames):
    # video_frames is now a list of NumPy arrays.
    grayscale_frames = []
    for frame in video_frames:
        # If the frame has one channel, remove it to get a 2D array; otherwise, transpose to get [height, width, channels]
        if frame.shape[0] == 1:
            frame = frame.squeeze(0)
        else:
            frame = np.transpose(frame, (1, 2, 0))
        image = Image.fromarray(frame)
        grayscale_image = image.convert("L")
        # Undo the conversion: convert the PIL grayscale image back to a NumPy array
        np_frame = np.array(grayscale_image)
        # Transpose back by adding a channel dimension to form [channels, height, width]
        np_frame = np_frame[np.newaxis, ...]
        
        grayscale_frames.append(np_frame)
    return grayscale_frames

# TODO Implement random crop of num_frames instead of center crop 
def adjust_frames(frames, num_frames):
    # Adjust the number of frames to match num_frames (pad or truncate as needed)
    if len(frames) < num_frames:
        frames = frames * (num_frames // len(frames)) + frames[:num_frames % len(frames)]
    else:
        frames = frames[:num_frames]
    return frames


## Metric Computation Utilities
def compute_mae(predictions, labels):
    return torch.mean(torch.abs(predictions - labels)).item()

def compute_std(predictions):
    return torch.std(predictions).item()

def compute_mse(predictions, labels):
    return torch.mean((predictions - labels) ** 2).item()

def compute_r2(predictions, labels):
    metric = R2Score()
    # metric.to("cuda")
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
