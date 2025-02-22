import argparse

def parse_args():
    """Parses command-line arguments with default values based on provided JSON config."""
    parser = argparse.ArgumentParser(
        description="Training script with configurable arguments."
    )

    parser.add_argument(
        "--contrastive_sigma",
        type=float,
        default=None,
        help="Sigma for contrastive kernelized loss"
    )

    parser.add_argument(
        "--kernel_type",
        type=str,
        default=None,
        help="Type of kernel for contrastive loss."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the training run."
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        default=None,
        help="Whether to resume training from a checkpoint."
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="Path to the dataset folder."
    )

    parser.add_argument(
        "--remove_unused_columns",
        action="store_true",
        default=None,
        help="Whether to remove unused columns."
    )

    parser.add_argument(
        "--resize_to",
        type=int,
        help="Resize images to this dimension."
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=None,
        help="Enable mixed precision (FP16) training."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for training."
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of warmup steps."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay for optimizer."
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Training batch size per device."
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        help="Evaluation batch size per device."
    )

    parser.add_argument(
        "--logging_first_step",
        action="store_true",
        default=None,
        help="Log the first step."
    )

    parser.add_argument(
        "--logging_strategy",
        type=str,
        choices=["steps", "epoch"],
        help="Logging strategy."
    )

    parser.add_argument(
        "--save_strategy",
        type=str,
        choices=["steps", "epoch"],
        help="Checkpoint saving strategy."
    )

    parser.add_argument(
        "--eval_strategy",
        type=str,
        choices=["steps", "epoch"],
        help="Evaluation strategy."
    )

    parser.add_argument(
        "--report_to",
        type=str,
        help="Reporting tool for logging (e.g., wandb, tensorboard)."
    )

    return parser.parse_args()

# Function that takes in input args and configs and overwrite config key if arg key is not None
def update_config(config, args):
    """Updates the configuration dictionary with command-line arguments."""
    for key, value in vars(args).items():
        if value is not None:
            key = f'{key}'
            config[key] = value
    return config