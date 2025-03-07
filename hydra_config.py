import os
import json
from omegaconf import DictConfig, OmegaConf

def auto_name_from_yaml(cfg: DictConfig, base_dir="."):
    """
    Generate an auto run name using the training_mode and image_size from model_config.
    
    This function uses the original logic of:
      - Shortening the training_mode via a mapping.
      - Building a base name as "<short_mode>_sz<image_size>_".
      - Scanning the base_dir for existing run names with the same prefix and 
        appending a numeric suffix (max + 1).
    
    Parameters:
        cfg (DictConfig): The Hydra configuration object, containing 'model_config'
                          and 'trainer_config'.
        base_dir (str): Directory where existing experiment folders are located.
    
    Returns:
        A string with the generated run name.
    """
    # Ensure required model config fields exist.
    if not (cfg.trainer_config or cfg.model_config) or "training_mode" not in cfg.trainer_config or "image_size" not in cfg.model_config:
        raise ValueError("model_config must include 'training_mode' and 'image_size' for auto naming.")
    
    training_mode = cfg.trainer_config.training_mode
    # Map verbose training modes to shorter acronyms.
    mode_map = {
        'regression': 'reg',
        'contrastive': 'con',
        'end_to_end_regression': 'rege2e',
        'end_to_end_contrastive': 'cone2e',
    }
    short_mode = mode_map.get(training_mode, training_mode)
    image_size = cfg.model_config.image_size
    learning_rate = repr(cfg.trainer_config.learning_rate)
    contrastive_sigma = repr(cfg.trainer_config.contrastive_sigma)
    weight_decay = repr(cfg.trainer_config.weight_decay)

    # Create the base name using the short training mode and image size.
    base_name = f"{short_mode}_sz{image_size}_lr{learning_rate}_cs{contrastive_sigma}_wd{weight_decay}_"
    max_suffix = 0
    
    # Look for directories (or files) in base_dir that start with base_name.
    for run in os.listdir(base_dir):
        if run.startswith(base_name):
            parts = run.split("_")
            try:
                # The suffix is assumed to be the last part after splitting.
                suffix = int(parts[-1])
                max_suffix = max(max_suffix, suffix)
            except ValueError:
                continue

    return f"{base_name}{max_suffix + 1}"

def update_experiment_name(cfg: DictConfig, base_dir="."):
    """
    Update the trainer's experiment_name field if it's set to "auto" using the auto naming logic.
    
    Parameters:
        cfg (DictConfig): The Hydra configuration object containing both 'trainer_config'
                          and 'model_config'.
        base_dir (str): The directory in which to search for existing run names.
    
    Returns:
        The updated configuration with the new experiment_name.
    """
    if cfg.experiment_name == "auto":
        cfg.experiment_name = auto_name_from_yaml(cfg, base_dir=base_dir)
    return cfg

def write_configs_to_json(cfg: DictConfig,
                          experiment_dir: str,
                          model_json_filename: str = "model_config.json",
                          trainer_json_filename: str = "trainer_config.json"):
    """
    Creates the experiment directory and writes out two JSON files extracted from the Hydra config.
    
    Parameters:
      cfg: The Hydra configuration object containing 'model_config' and 'trainer_config' groups.
      experiment_dir: The directory where the JSON files will be written.
      model_json_filename: Filename for the model configuration JSON (default: "model_config.json").
      trainer_json_filename: Filename for the trainer configuration JSON (default: "trainer_config.json").
    
    Returns:
      A tuple (model_json_path, trainer_json_path) with the paths to the written JSON files.
    """
    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Define full paths for the JSON files
    model_json_path = os.path.join(experiment_dir, model_json_filename)
    trainer_json_path = os.path.join(experiment_dir, trainer_json_filename)
    
    # Write the model_config group to a JSON file
    with open(model_json_path, "w") as model_file:
        json.dump(OmegaConf.to_container(cfg.model_config, resolve=True), model_file, indent=2)
    
    # Write the trainer_config group to a JSON file
    with open(trainer_json_path, "w") as trainer_file:
        json.dump(OmegaConf.to_container(cfg.trainer_config, resolve=True), trainer_file, indent=2)
    
    return model_json_path, trainer_json_path
