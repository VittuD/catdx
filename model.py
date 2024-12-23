from transformers import VivitForVideoClassification
import torch.nn as nn

class VivitWithOptionalProjectionHead(nn.Module):
    """
    Wrapper class for Vivit model with a custom projection head.
    """
    def __init__(self, model_name, config, projection_dim=128, add_projection_head=True):
        super().__init__()
        # Load the base ViViT model
        self.vivit = VivitForVideoClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            config=config,
            ignore_mismatched_sizes=True
        )
        self.add_projection_head = add_projection_head

        # Add projection head if specified
        if self.add_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, projection_dim)
            )
        else:
            self.projection_head = None
    
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

        print(projections.shape)
        # Return both logits and projections
        return {"logits": outputs.logits, "projections": projections}

def load_model(config, model_name, projection_dim=128, add_projection_head=True):
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
    model = VivitWithOptionalProjectionHead(
        model_name, config, projection_dim, add_projection_head
    )

    if hasattr(config, "freeze_backbone") and config.freeze_backbone:
        freeze_backbone(model.vivit)
        print("Backbone frozen.")

    return model


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
        else:
            param.requires_grad = True
