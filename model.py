from transformers import VivitForVideoClassification

def load_model(config, model_name):
    """
    Load a pretrained video classification model with the given configuration.

    Args:
        config (dict): Configuration dictionary containing the model's configuration details.

    Returns:
        model: A VivitForVideoClassification instance.
    """
    model = VivitForVideoClassification.from_pretrained(
        pretrained_model_name_or_path=model_name, 
        config=config,
        ignore_mismatched_sizes=True
    )

    if hasattr(config, "freeze_backbone") and config.freeze_backbone:
        freeze_backbone(model)
        print("Backbone frozen.")

    return model

def freeze_backbone(model):
    """
    Freeze the backbone of the model.

    Args:
        model: The model to freeze.
    """
    for name, param in model.named_parameters():
        # Freeze everything that isn't part of the classifier
        if "classifier" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
