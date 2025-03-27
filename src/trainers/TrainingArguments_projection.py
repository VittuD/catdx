from dataclasses import dataclass, field
from transformers import TrainingArguments


# Custom TrainingArguments class to condense every argument into a single file
# Let's use https://huggingface.co/docs/transformers/internal/trainer_utils#transformers.HfArgumentParser
# https://stackoverflow.com/questions/75786601/how-to-convert-trainingarguments-object-into-a-json-file
# https://github.com/huggingface/transformers/issues/9912
@dataclass
class TrainingArguments_projection(TrainingArguments):
    """
    Custom TrainingArguments that extend the HuggingFace TrainingArguments with additional
    parameters for use in the custom trainer.

    Additional parameters:
        kernel_type (str): Kernel type for the custom trainer (default: "gaussian").
        contrastive_sigma (float): Sigma value for contrastive learning (default: 1.0).
        training_mode (str): Training mode (default: "regression").
    """
    kernel_type: str = field(default="gaussian", metadata={"help": "Kernel type for the custom trainer."})
    contrastive_sigma: float = field(default=1.0, metadata={"help": "Sigma value for contrastive learning."})
    training_mode: str = field(default="regression", metadata={"help": "Training mode to be used."})
    dataset_folder: str = field(default="dataset", metadata={"help": "Folder name of the dataset."})
    contrastive_method: str = field(default="supcon", metadata={"help": "Contrastive method to be used."})
    is_unsupervised: bool = field(default=False, metadata={"help": "Whether the training is unsupervised."})
    gather_loss: bool = field(default=False, metadata={"help": "Whether to gather predictions before computing the loss. (Works for multiple GPUs setup)"})

    def pop_attribute(self, attribute, default=None):
        """
        Pop (gets and deletes) the selected attribute from the TrainingArguments object and returns it's value or default.
        """
        if hasattr(self, attribute):
            value = getattr(self, attribute)
            delattr(self, attribute)
            return value
        return default
    