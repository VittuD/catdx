import torch.nn as nn
from modeling_vivit_projection import VivitWithOptionalProjectionHead

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

    # Debug: print each parameter's trainability
    for name, param in model.named_parameters():
        print(f'{name}: {param.requires_grad}')

    return model
