from transformers import VivitForVideoClassification, VivitConfig
import torch.nn as nn

class VivitWithOptionalProjectionHead(VivitForVideoClassification):
    """
    Wrapper class for the Vivit model with a custom projection head.
    """
    config_class = VivitConfig

    def __init__(self, config: VivitConfig, projection_dim=128, add_projection_head=True):
        super().__init__(config)
        self.add_projection_head = add_projection_head

        # Add projection head if specified
        if self.add_projection_head:
            if not hasattr(self.vivit, 'projection_head'):
                self.vivit.projection_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, projection_dim)
                )
        else:
            self.vivit.projection_head = None

    def forward(self, pixel_values, **kwargs):
        # Forward pass through the base model
        outputs = super().forward(pixel_values=pixel_values, output_hidden_states=True, **kwargs)
        hidden_states = outputs.hidden_states  # Hidden states from all transformer layers
        last_hidden_state = hidden_states[-1]    # Last hidden state (batch_size, seq_len, hidden_size)

        # Apply projection head if specified
        projections = None
        if self.add_projection_head or self.vivit.projection_head is not None:
            # Use the [CLS] token embedding for projection
            cls_token = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
            projections = self.vivit.projection_head(cls_token)

        return {"logits": outputs.logits, "projections": projections, "hidden_states": hidden_states}
