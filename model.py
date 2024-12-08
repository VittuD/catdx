from transformers import VivitForVideoClassification

def load_model(config, model_name):
    """
    Load a pretrained video classification model with the given configuration.

    Args:
        config (dict): Configuration dictionary containing the model's configuration details.

    Returns:
        model: A VivitForVideoClassification instance.
    """
    return VivitForVideoClassification.from_pretrained(
        pretrained_model_name_or_path=model_name, 
        config=config,
        ignore_mismatched_sizes=True
    )
