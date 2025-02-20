from transformers import TrainingArguments, VivitConfig
import json
import os

# Load configuration from a JSON file
def load_config(config_path='config.json', vivit_config=None):
    with open(config_path, 'r') as f:
        base_config = json.load(f)
        # If run name is 'auto' generate it
        if base_config.get('run_name') == 'auto':
            base_config['run_name'] = auto_name(base_config, vivit_config)
        return base_config

def auto_name(config, vivit_config):
    if vivit_config is None:
        raise ValueError('vivit_config must be provided if run_name is set to "auto"')
    
    training_mode = vivit_config.training_mode
    base_name = f"{training_mode}_"
    max_suffix = 0  # Start at zero so that if no runs exist, new suffix will be 1.
    
    for run in os.listdir('.'):
        if run.startswith(base_name):
            parts = run.split("_")
            try:
                suffix = int(parts[-1])
                max_suffix = max(max_suffix, suffix)
            except ValueError:
                # If the suffix isn't an integer, skip this run.
                continue
    # Append the next integer (max_suffix + 1) to the base name.
    return f"{base_name}{max_suffix + 1}"
    
def get_training_args(config):
    return TrainingArguments(
        output_dir=config.get('run_name'),
        report_to=config.get('report_to', 'wandb'),
        learning_rate=config.get('learning_rate', 5e-5),
        warmup_steps=config.get('warmup_steps', 0),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 8),
        num_train_epochs=config.get('num_train_epochs', 5),
        weight_decay=config.get('weight_decay', 0.01),
        logging_first_step=config.get('logging_first_step', True),
        logging_strategy=config.get('save_strategy', 'epoch'),
        save_strategy=config.get('save_strategy', 'epoch'),
        eval_strategy=config.get('eval_strategy', 'epoch'),
        fp16=config.get('fp16', True),
        remove_unused_columns=config.get('remove_unused_columns', False),
        resume_from_checkpoint=config.get('resume_from_checkpoint', False),
        seed=42,
    )

# Main for testing
def main():
    # Load model configuration
    model_config = 'model_config.json'
    model_config = VivitConfig.from_json_file(model_config)
    print(model_config)

    # Load training configuration
    config = load_config('config.json')
    training_args = get_training_args(config)
    print(training_args)


if __name__ == '__main__':
    main()
    