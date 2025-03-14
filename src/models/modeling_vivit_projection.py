from transformers import VivitForVideoClassification, VivitConfig
import torch.nn as nn

class VivitWithOptionalProjectionHead(VivitForVideoClassification):
    """
    Wrapper class for the Vivit model with a custom projection head.
    Includes helper methods to freeze components based on the training mode.
    """
    config_class = VivitConfig

    def __init__(self, config: VivitConfig, projection_dim=128, add_projection_head=True):
        super().__init__(config)
        self.add_projection_head = add_projection_head
        self.training_mode = None  # will be set via set_training_mode()

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

        # Freeze components present in VivitConfig.freeze list
        if hasattr(config, 'freeze'):
            if not hasattr(self, "freeze"):
                self.freeze = []
            for component in config.freeze:
                print(f"Vivit Model : Freezing {component}")
                self.freeze.append(component)
                self.freeze_component(component)

    def forward(self, pixel_values, **kwargs):
        # Forward pass through the base model
        outputs = super().forward(pixel_values=pixel_values, output_hidden_states=True, **kwargs)
        hidden_states = outputs.hidden_states  # Hidden states from all transformer layers
        last_hidden_state = hidden_states[-1]    # Last hidden state (batch_size, seq_len, hidden_size)

        # Apply projection head if specified
        projections = None
        if self.add_projection_head and self.vivit.projection_head is not None:
            # Use the [CLS] token embedding for projection
            cls_token = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
            projections = self.vivit.projection_head(cls_token)

        return {"logits": outputs.logits, "projections": projections, "hidden_states": hidden_states}

    @property
    def backbone_parameters(self):
        """
        Returns all parameters that are part of the backbone—that is,
        those that are not in the classifier or projection head.
        """
        return (
            param 
            for name, param in self.named_parameters() 
            if "classifier" not in name and "projection_head" not in name
        )

    def freeze_component(self, component: str):
        """
        Freeze parameters of a given component of the model.

        Args:
            component (str): The component to freeze. Options include:
                - "backbone": Freezes all parameters of the main Vivit model.
                - "projection_head": Freezes parameters in the projection head.
                - "classifier": Freezes parameters in the classifier head.
        """
        if component == "backbone":
            for param in self.backbone_parameters:
                param.requires_grad = False
        elif component == "projection_head":
            if self.vivit.projection_head is not None:
                for param in self.vivit.projection_head.parameters():
                    param.requires_grad = False
        elif component == "classifier":
            # Assuming the classifier is defined in the parent class.
            if hasattr(self, "classifier"):
                for param in self.classifier.parameters():
                    param.requires_grad = False

    def set_training_mode(self, training_mode: str):
        """
        Configure the model for a specific training mode by freezing components.

        For example:
            - "contrastive": Freezes the classifier.
            - "regression": Freezes the backbone and projection head.
            - Modes starting with "end_to_end_" are normalized.

        Args:
            training_mode (str): The training mode to set.
        """
        if "end_to_end" in training_mode:
            self.training_mode = training_mode
            # End-to-end mode: no freezing.
            return
        else:
            self.training_mode = training_mode

        # Define mapping from training mode to components to freeze.
        freeze_map = {
            "contrastive": ["classifier"],
            "regression": ["backbone", "projection_head"]
        }

        if training_mode not in freeze_map:
            print("Training mode unknown. Defaulting to 'regression'.")
            training_mode = "regression"
            self.training_mode = training_mode

        if not hasattr(self, "freeze"):
            self.freeze = []

        for component in freeze_map[training_mode]:
            if component not in self.freeze:
                self.freeze.append(component)
                self.freeze_component(component)
