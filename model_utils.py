import torch.nn as nn
from vivit_model import VivitWithOptionalProjectionHead

def load_model(vivit_config, is_pretrained=False, projection_dim=128, add_projection_head=True):
    """
    Load the Vivit model with an optional projection head.

    Args:
        vivit_config: Configuration object with model details.
        is_pretrained (bool): Whether to load pretrained weights.
        projection_dim (int): Dimension of the projection head.
        add_projection_head (bool): Flag to add the projection head.

    Returns:
        model: An instance of VivitWithOptionalProjectionHead.
    """
    if is_pretrained:
        model = VivitWithOptionalProjectionHead.from_pretrained(
            pretrained_model_name_or_path=vivit_config.model_name_or_path,
            config=vivit_config,
            ignore_mismatched_sizes=True
        )
    else:
        model = VivitWithOptionalProjectionHead(vivit_config, projection_dim, add_projection_head)

    # Unfreeze all parameters for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Handle training mode and freeze specified components
    handle_training_mode(vivit_config)
    print(f"Training mode: {vivit_config.training_mode}")

    if hasattr(vivit_config, "freeze") and isinstance(vivit_config.freeze, list):
        print(f"Freezing: {vivit_config.freeze}")
        for element in vivit_config.freeze:
            freeze_element(model, element)

    # Debug: print each parameter's trainability
    for name, param in model.named_parameters():
        print(f'{name}: {param.requires_grad}')

    return model

def freeze_element(model, element):
    """
    Freeze a specific element of the model.
    
    Args:
        model: The model to modify.
        element: Name substring of the element to freeze, or "backbone" for the backbone.
    """
    if element == "backbone":
        freeze_backbone(model)
    else:
        for name, param in model.named_parameters():
            if element in name:
                param.requires_grad = False

def freeze_backbone(model):
    """
    Freeze the backbone of the model.
    
    Args:
        model: The model to modify.
    """
    for name, param in model.named_parameters():
        # Freeze everything that is not part of the classifier or projection head
        if "classifier" not in name and "projection_head" not in name:
            param.requires_grad = False

def reinitialize_classifier_heads(model, vivit_config):
    """
    Reinitialize classifier heads with random weights.
    
    Args:
        model: The model whose classifier heads will be reinitialized.
        vivit_config: Configuration object with 'num_labels' and 'hidden_size'.
    """
    if hasattr(vivit_config, "num_labels") and vivit_config.num_labels > 0:
        model.classifier = nn.Linear(vivit_config.hidden_size, vivit_config.num_labels)
        model.classifier.reset_parameters()

def handle_training_mode(vivit_config):
    """
    Handle training mode adjustments and component freezing based on the configuration.
    
    Args:
        vivit_config: Configuration object with training mode and freeze list.
    """
    if not hasattr(vivit_config, "training_mode"):
        return

    mode = vivit_config.training_mode

    # Normalize end-to-end modes immediately.
    if mode.startswith("end_to_end_"):
        vivit_config.training_mode = mode.split("_")[-1]
        return

    # Map training modes to components to freeze.
    freeze_map = {
        "contrastive": ["classifier"],
        "regression": ["backbone", "projection_head"]
    }

    # Default to 'regression' if an unknown mode is encountered.
    if mode not in freeze_map:
        print("Training mode set to default: regression")
        vivit_config.training_mode = "regression"
        mode = "regression"

    for component in freeze_map[mode]:
        if component not in vivit_config.freeze:
            vivit_config.freeze.append(component)
