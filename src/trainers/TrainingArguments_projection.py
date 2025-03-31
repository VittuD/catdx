from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class TrainingArguments_projection(TrainingArguments):
    """
    Custom TrainingArguments that extend the HuggingFace TrainingArguments with additional
    parameters for use in the custom trainer.
    """
    kernel_type: str = field(default="gaussian", metadata={"help": "Kernel type for the custom trainer."})
    contrastive_sigma: float = field(default=1.0, metadata={"help": "Sigma value for contrastive learning."})
    training_mode: str = field(default="regression", metadata={"help": "Training mode to be used."})
    dataset_folder: str = field(default="dataset", metadata={"help": "Folder name of the dataset."})
    contrastive_method: str = field(default="supcon", metadata={"help": "Contrastive method to be used."})
    is_unsupervised: bool = field(default=False, metadata={"help": "Whether the training is unsupervised."})
    gather_loss: bool = field(default=False, metadata={"help": "Whether to gather predictions before computing the loss."})
    # Override the lr_scheduler_type field to be a plain string.
    lr_scheduler_type: str = field(
        default="constant_with_warmup",
        metadata={
            "help": (
                "Type of learning rate scheduler. Allowed values include: "
                "linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, "
                "inverse_sqrt, reduce_lr_on_plateau, cosine_with_min_lr, warmup_stable_decay, or a custom value (e.g., 'step')."
            )
        },
    )

    def __post_init__(self):
        # TODO quick and dirty fix to bypass the Enum conversion in the parent class
        # Save the original value for lr_scheduler_type (e.g. "step")
        custom_lr_scheduler_type = self.lr_scheduler_type
        # Temporarily set to a valid value to bypass the Enum conversion in the parent.
        self.lr_scheduler_type = "constant_with_warmup"
        super().__post_init__()
        # Restore your custom scheduler type.
        self.lr_scheduler_type = custom_lr_scheduler_type
