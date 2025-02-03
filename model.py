from transformers import VivitForVideoClassification, VivitConfig
import torch.nn as nn


## TODO freeze the regression head when doing contrastive pretraining


class VivitWithOptionalProjectionHead(VivitForVideoClassification):
    """
    Wrapper class for Vivit model with a custom projection head.
    """
    config_class = VivitConfig

    def __init__(self, config: VivitConfig, projection_dim=128, add_projection_head=True):
        
        super().__init__(config)
        # Load the base ViViT model
        self.vivit = VivitForVideoClassification.from_pretrained(
            pretrained_model_name_or_path=config.model_name_or_path, 
            config=config,
            ignore_mismatched_sizes=True
        )
        self.add_projection_head = add_projection_head

        # Add projection head if specified and not already present
        if self.add_projection_head:
            if not hasattr(self.vivit, 'projection_head'):
                self.projection_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, projection_dim)
                )
            else:
                self.projection_head = self.vivit.projection_head
        else:
            self.projection_head = None
    
    # def __init__(self, config):


    ### hidden states -> | projection head | -> regression head -> logits


    def forward(self, pixel_values, **kwargs):
        # Forward pass through the base model
        outputs = self.vivit(pixel_values=pixel_values, output_hidden_states=True, **kwargs)
        
        # Extract hidden states
        hidden_states = outputs.hidden_states  # List of hidden states from all transformer layers

        # Optionally get the last hidden state
        last_hidden_state = hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)

        # Apply projection head if specified
        projections = None

        if self.add_projection_head and self.projection_head is not None:
            # Optionally pool the sequence embeddings (mean pooling or [CLS] token)
            pooled_output = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)
            projections = self.projection_head(pooled_output)  # Shape: (batch_size, projection_dim)

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

    # For contrastive pretraining, freeze the classifier head automatically
    if hasattr(vivit_config, "vivit_training_mode") and vivit_config.vivit_training_mode == "contrastive":
        print(f'Adding classifier head to the Freeze list for contrastive pretraining:')
        # Add classifier to the list of elements to freeze if not already present
        if "classifier" not in vivit_config.freeze:
            vivit_config.freeze.append("classifier")

    # For regression training, freeze the backbone and projection head
    if hasattr(vivit_config, "vivit_training_mode") and vivit_config.vivit_training_mode == "regression":
        print(f'Adding backbone and projection head to the Freeze list for regression training:')
        # Add backbone and projection head to the list of elements to freeze if not already present
        if "backbone" not in vivit_config.freeze:
            vivit_config.freeze.append("backbone")
        if "projection_head" not in vivit_config.freeze:
            vivit_config.freeze.append("projection_head")

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
