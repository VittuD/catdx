# trainer.py

from typing import Union
from src.trainers.TrainingArguments_projection import TrainingArguments_projection
from transformers import Trainer, TrainingArguments, PreTrainedModel
from src.utils.utils import compute_r2, compute_mae, compute_std, compute_mse
from scipy.stats import pearsonr
import torch
from torch import nn
from src.losses.contrastive import kernelized_supcon_loss
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO only 1 forward with bsz*2 and then torch split
# TODO if unsupervised pass labels,kernel = None and method = 'yaware'

class LogTrainer(Trainer):
        
    def __init__(self,
                model: Union[PreTrainedModel, nn.Module] = None,
                args: TrainingArguments_projection = None,
                **kwargs):

        # Get attributes from args
        self.training_mode = args.training_mode
        self.kernel_type = args.kernel_type
        self.contrastive_sigma = args.contrastive_sigma
        self.contrastive_method = args.contrastive_method
        self.is_unsupervised = args.is_unsupervised

        model.set_training_mode(self.training_mode)

        # Halve the batch size for unsupervised learning (since we double the batch size with augmentations)
        if self.is_unsupervised:
            args.per_device_train_batch_size //= 2
            args.per_device_eval_batch_size //= 2

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
        """
        Compute the training loss for both supervised and unsupervised modes.

        Parameters:
            model: the neural network model
            inputs: dictionary of input tensors (must include 'labels' and 'pixel_values')
            num_items_in_batch: number of items in the batch to process for augmentation
            return_outputs: if True, returns a tuple (loss, outputs), else only loss

        Returns:
            loss (and optionally model outputs)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        features = outputs.get("projections")

        if self.is_unsupervised:
            outputs, features, labels = self._process_unsupervised(model, inputs, num_items_in_batch, features)

        loss = self._compute_loss(outputs, features, labels, model)
        self._log_epoch_wise_predictions(outputs, labels)

        return (loss, outputs) if return_outputs else loss

    def _process_unsupervised(self, model, inputs, num_items_in_batch, features):
        """
        Process unsupervised training by augmenting the inputs and combining features.

        Returns:
            outputs: the model outputs for the augmented inputs
            features: combined features tensor with shape [bsz, n_views, n_features]
            labels: set to None for unsupervised learning
        """
        augmented_inputs = self._augment_inputs(inputs, num_items_in_batch)
        augmented_outputs = model(**augmented_inputs)
        augmented_features = augmented_outputs.get("projections")

        # Concatenate original and augmented features to create two views per sample
        features = torch.cat((features.unsqueeze(1), augmented_features.unsqueeze(1)), dim=1)
        labels = None
        return augmented_outputs, features, labels
    
    def _get_names(self, obj, scope):
        # Returns a list of variable names in the provided scope that refer to obj.
        return [name for name, value in scope.items() if value is obj]

    def _augment_inputs(self, inputs, num_items_in_batch):
        """
        Create an augmented version of the inputs for unsupervised learning.

        Parameters:
             inputs: original input dictionary (expects 'pixel_values')
             num_items_in_batch: number of items to augment

        Returns:
             A new dictionary with augmented inputs.
        """
        augmented_inputs = {k: v.clone() for k, v in inputs.items()}

        for i in range(num_items_in_batch):
            video = augmented_inputs['pixel_values'][i]
            transformed_frames = []
            for j in range(video.shape[0]):
                # Convert frame from (C, H, W) to (H, W, C) numpy array
                frame = video[j].permute(1, 2, 0).cpu().numpy()
                aug_frame = self._apply_augmentations(frame)
                # Convert back to torch tensor with shape (C, H, W)
                aug_tensor = torch.tensor(aug_frame).permute(2, 0, 1)
                transformed_frames.append(aug_tensor)

                # Save the image for manual sanity check
                # original_filename = f'frame_{j}_original.png'
                # augmented_filename = f'frame_{j}_augmented.png'
                # self._debug_save_frame(frame, permute=False, eq_filename=original_filename)
                # self._debug_save_frame(aug_frame, permute=False, eq_filename=augmented_filename)

            augmented_inputs['pixel_values'][i] = torch.stack(transformed_frames)
        return augmented_inputs

    def _apply_augmentations(self, frame):
        """
        Apply a series of augmentations to a single frame.

        The transformations include converting to grayscale, applying random brightness/contrast,
        and performing a random resized crop.

        Parameters:
             frame: numpy array of shape (H, W, C)

        Returns:
             The augmented frame as a numpy array.
        """
        # It's bc albumentation accept normalize to either uint8 images: [0, 255] or float32 images: [0, 1]
        transform = A.Compose([
            A.ToGray(always_apply=True),
            A.RandomBrightnessContrast(p=1, brightness_limit=0.25, contrast_limit=0.25),
            A.RandomResizedCrop(size=(frame.shape[0], frame.shape[1]), scale=(0.6, 0.9)),
        ])
        augmented = transform(image=frame)
        return augmented['image']

    def _compute_loss(self, outputs, features, labels, model):
        """
        Compute the loss using either MSE or kernelized supervised contrastive loss.

        Parameters:
             outputs: model outputs
             features: tensor of projection features (if available)
             labels: ground truth labels (None for unsupervised training)
             model: model instance with training mode information

        Returns:
             The computed loss.
        """
        use_mse = features is None or 'regression' in model.training_mode
        if features is None:
            print("Warning: 'projections' not found. Defaulting to regression (MSE loss), which may be unintended.")

        if use_mse:
            return self.mse_loss(outputs, labels)
        else:
            # If no additional views are present, add a singleton dimension
            if features.dim() == 2:
                features = features.unsqueeze(1)
            return kernelized_supcon_loss(
                features=features,  # Expected shape: [bsz, n_views, n_features]
                labels=labels,
                kernel_type=self.kernel_type,    
                temperature=0.07,
                sigma=self.contrastive_sigma,
                method=self.contrastive_method,                  
                contrast_mode='all',
                base_temperature=0.07,
                delta_reduction='sum'
            )

    def _log_epoch_wise_predictions(self, outputs, labels):
        """
        Log predictions and labels for epoch-wise evaluation metrics (e.g., R²).

        Parameters:
             outputs: model outputs containing logits
             labels: ground truth labels
        """
        predictions = outputs['logits'].squeeze()
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)

        self.epoch_wise_predictions = torch.cat(
            (self.epoch_wise_predictions, predictions.detach().cpu())
        )
        # Stub the labels tensor if it is None (unsupervised learning)
        if labels is None:
            labels = torch.full_like(predictions, float("nan"))
        self.epoch_wise_labels = torch.cat(
            (self.epoch_wise_labels, labels.detach().cpu())
        )

    def _debug_save_frame(self, frame, permute=True, eq_filename='first_frame_eq.png', hist_filename='histogram.png'):
        """
        Processes a frame tensor by normalizing, applying histogram equalization,
        and saving both the equalized image and a histogram plot.

        Parameters:
            frame (torch.Tensor): frame tensor of shape [C, H, W] (e.g., where C is 1)
            eq_filename (str): Filename for the equalized image to save.
            hist_filename (str): Filename for the histogram image to save.
        """
        # If frame is not a tensor, convert it to one
        if not torch.is_tensor(frame):
            frame = torch.tensor(frame)
        # Convert the frame to a numpy array and remove the channel dimension if it is 1.
        if permute:
            frame = frame.permute(1, 2, 0)  # shape (H, W, C)
        if frame.shape[2] == 1:
            frame = frame.squeeze(axis=2)  # shape (H, W)

        frame = frame.cpu().numpy()

        # Normalize to [0, 255]
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min()) * 255
        frame_norm = frame_norm.astype(np.uint8)

        # Apply histogram equalization
        frame_eq = cv2.equalizeHist(frame_norm)

        # Save the equalized frame image
        cv2.imwrite(eq_filename, frame_eq)

        # Save the histogram plot
        plt.hist(frame_norm.ravel(), bins=256, range=[0,255])
        plt.title('Pixel Intensity Distribution')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.savefig(hist_filename)
        plt.close()
