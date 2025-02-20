from transformers import VivitForVideoClassification, VivitConfig
import torch.nn as nn


class VivitWithOptionalProjectionHead(VivitForVideoClassification):
    """
    Wrapper class for Vivit model with a custom projection head.
    """
    config_class = VivitConfig

    def __init__(self, config: VivitConfig, projection_dim=128, add_projection_head=True):
        
        super().__init__(config)

        self.add_projection_head = add_projection_head

        # Add projection head if specified and not already present
        if self.add_projection_head:
            if not hasattr(self.vivit, 'projection_head'):
                self.vivit.projection_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, projection_dim)
                )
        else:
            self.vivit.projection_head = None

    ### hidden states -> | projection head | -> regression head -> logits


    def forward(self, pixel_values, **kwargs):
        # Forward pass through the base model
        outputs = super().forward(pixel_values=pixel_values, output_hidden_states=True, **kwargs)
        
        # Extract hidden states
        hidden_states = outputs.hidden_states  # List of hidden states from all transformer layers

        # Optionally get the last hidden state
        last_hidden_state = hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)

        # Apply projection head if specified
        projections = None

        if self.add_projection_head and self.vivit.projection_head is not None:
            # Optionally pool the sequence embeddings (mean pooling or [CLS] token)
            # pooled_output = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)
            # projections = self.vivit.projection_head(pooled_output)  # Shape: (batch_size, projection_dim)

            cls_token = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
            projections = self.vivit.projection_head(cls_token)  # Shape: (batch_size, projection_dim)

        # Return both logits and projections
        return {"logits": outputs.logits, "projections": projections, "hidden_states": hidden_states}

def load_model(vivit_config, is_pretrained=False, projection_dim=128, add_projection_head=True):
    """
    Load the Vivit model with an optional projection head.

    Args:
        config (dict): Configuration dictionary containing the model's configuration details.
        model_name (str): The pretrained model name.
        projection_dim (int): The dimension of the projection head.
        add_projection_head (bool): Whether to include a projection head.

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
        model = VivitWithOptionalProjectionHead(
            vivit_config, projection_dim, add_projection_head
        )

    # Unfreeze everything preemptively to allow for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Re-initialize the classifier heads with random weights
    # reinitialize_classifier_heads(model, vivit_config)

    # Handle the training mode and freeze the specified components
    handle_training_mode(vivit_config)
    print(f"Training mode: {vivit_config.vivit_training_mode}")

    if hasattr(vivit_config, "freeze") and isinstance(vivit_config.freeze, list):
        print(f"Freezing: {vivit_config.freeze}")
        for element in vivit_config.freeze:
            freeze_element(model, element)

    # Debug print each layer with if it requires_grad or not
    for name, param in model.named_parameters():
        print(f'{name}: {param.requires_grad}')

    return model

def freeze_element(model, element):
    """
    Freeze a specific element of the model.

    Args:
        model: The model to freeze.
        element: The element to freeze.
    """
    for name, param in model.named_parameters():
        if element == "backbone":
            freeze_backbone(model)
        else:
            if element in name:
                param.requires_grad = False

def freeze_backbone(model):
    """
    Freeze the backbone of the model.

    Args:
        model: The model to freeze.
    """
    for name, param in model.named_parameters():
        # Freeze everything that isn't part of the classifier or projection head
        if "classifier" not in name and "projection_head" not in name:
            param.requires_grad = False


def reinitialize_classifier_heads(model, vivit_config):
    """
    Re-initialize both classifier heads with random weights for:
    "vivit.classifier.weight", "vivit.classifier.bias",
    "classifier.weight", and "classifier.bias".
    Args:
        model: The model whose classifier heads will be re-initialized.
        vivit_config: The configuration object containing num_labels and hidden_size.
    """
    if hasattr(vivit_config, "num_labels") and vivit_config.num_labels > 0:
        # Re-initialize the outer classifier head
        model.classifier = nn.Linear(vivit_config.hidden_size, vivit_config.num_labels)
        model.classifier.reset_parameters()

def handle_training_mode(vivit_config):
    """
    Handle the training mode and freeze the specified components.
    Args:
        vivit_config: The configuration object containing the training mode and freeze list.
    """
    if not hasattr(vivit_config, "vivit_training_mode"):
        return
    
    mode = vivit_config.vivit_training_mode

    # Normalize end-to-end modes and exit immediately.
    if mode.startswith("end_to_end_"):
        vivit_config.vivit_training_mode = mode.split("_")[-1]
        return
    
    # Mapping of modes to the components to freeze.
    freeze_map = {
        "contrastive": ["classifier"],
        "regression": ["backbone", "projection_head"]
    }

    # Default to regression if an unknown mode is encountered.
    if mode not in freeze_map:
        print("Training mode set to default: regression")
        vivit_config.vivit_training_mode = "regression"
        mode = "regression"

    for component in freeze_map[mode]:
        if component not in vivit_config.freeze:
            vivit_config.freeze.append(component)
