import datasets
import torch
from torcheval.metrics import R2Score
import numpy as np
from transformers import VivitImageProcessor, TrainerCallback
from torchvision.transforms import Compose, Normalize, RandomCrop, ColorJitter, ToTensor
import torchvision.transforms.functional as F
import random
import os

def get_image_processor(resize_to):
    return VivitImageProcessor(
        do_resize=True,
        size={'height': resize_to, 'width': resize_to},
        do_center_crop=False,
        do_normalize=True,
    )

# Dataset Utilities
def load_dataset(dataset_folder, augmentation=None):
    # Augmentation can be 'jitter', 'crop', 'cutout', 'all' or None
    dataset = datasets.load_dataset(dataset_folder)
    dataset = dataset.rename_column('video', 'pixel_values')
    dataset = dataset.rename_column('CO', 'labels')
    # For every augmentation, apply the corresponding transformation function to the dataset
    if augmentation == 'all':
        dataset.set_transform(transforms)
    return dataset

def get_frame_transform(first_frame):
    """
    Build a Compose pipeline for per-frame processing.
    
    This pipeline will:
      1. Apply a fixed color jitter (with random parameters sampled once).
      2. Apply a fixed random crop (here, we crop to 100x100).
      3. Pad the result back to 112x112 (i.e. 6 pixels on each side, since 100+6+6 = 112).
      4. Convert the frame to a tensor.
    """
    # Define the desired crop size (smaller than 112x112)
    crop_size = (100, 100)
    
    # Sample crop parameters from the first frame
    i, j, h, w = RandomCrop.get_params(first_frame, output_size=crop_size)
    crop_fn = lambda img: F.crop(img, i, j, h, w)
    
    # Pad 6 pixels on left, top, right, and bottom to bring the 100x100 crop back to 112x112
    pad_fn = lambda img: F.pad(img, padding=(6, 6, 6, 6))
    
    # Sample fixed color jitter parameters once (for brightness and hue)
    jitter_fn = ColorJitter.get_params(
        brightness=[0.8, 1.2],
        contrast=None,
        saturation=None,
        hue=[-0.1, 0.1]
    )
    
    # Build and return the Compose pipeline for each frame
    return Compose([
        lambda img: jitter_fn(img),  # Apply the fixed color jitter
        crop_fn,                     # Apply the fixed crop
        pad_fn,                      # Pad to 112x112
        ToTensor()                   # Convert to tensor
    ])

def transforms(example):
    """
    Process a video (list of PIL Images) to:
      - Apply consistent per-frame augmentations (color jitter, crop, pad)
      - Optionally apply cutout (zero out a patch) over the entire video tensor
      - Normalize the video
    The final video tensor is stored in example["pixel_values"] with shape [C, T, H, W].
    """
    # Here we assume the list of PIL Images is stored under "pixel_values"
    video = example["pixel_values"]
    if len(video) == 0:
        return example

    # Get a per-frame transform (with fixed randomness) using the first frame
    first_frame = video[0].convert("RGB")
    frame_transform = get_frame_transform(first_frame)
    
    # Apply the same transform to each frame to ensure consistency
    processed_frames = [frame_transform(frame.convert("RGB")) for frame in video]
    
    # Stack frames into a tensor of shape [T, C, H, W] then permute to [C, T, H, W]
    video_tensor = torch.stack(processed_frames, dim=0)  # shape: [T, C, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3)        # shape: [C, T, H, W]
    
    # With 50% probability, apply a cutout:
    # Zero out a patch of size [2, 16, 16] (i.e. one frame in time and a 32x32 spatial region)
    if random.random() < 0.5:
        C, T, H, W = video_tensor.shape  # expected: [3, 32, 112, 112]
        patch_t = 2
        patch_h = 16
        patch_w = 16
        t_start = random.randint(0, T - patch_t)
        h_start = random.randint(0, H - patch_h)
        w_start = random.randint(0, W - patch_w)
        video_tensor[:, t_start:t_start+patch_t, h_start:h_start+patch_h, w_start:w_start+patch_w] = 0.0
    
    # Normalize the video.
    # (Here using a no-op normalization; change mean and std as needed.)
    normalize = Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    normalized_frames = [normalize(video_tensor[:, t, :, :]) for t in range(video_tensor.shape[1])]
    video_tensor = torch.stack(normalized_frames, dim=1)  # back to shape [C, T, H, W]
    
    example["pixel_values"] = video_tensor
    return example

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

def generate_log_filename(pdf_file, run_name):
    filename = os.path.basename(pdf_file)
    alias = filename.split('_')[-1].split('.')[0]
    new_filename = f"{run_name}_predictions_report_{alias}.pdf"
    return new_filename
