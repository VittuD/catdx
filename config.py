from transformers import TrainingArguments, VivitConfig
import json

# Load configuration from a JSON file
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)
    
def get_training_args(config):
    return TrainingArguments(
        output_dir=config.get("run_name"),
        report_to=config.get("report_to", "wandb"),
        learning_rate=config.get("learning_rate", 5e-5),
        warmup_steps=config.get("warmup_steps", 0),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        num_train_epochs=config.get("num_train_epochs", 5),
        weight_decay=config.get("weight_decay", 0.01),
        logging_first_step=config.get("logging_first_step", True),
        logging_strategy=config.get("save_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        fp16=config.get("fp16", True),
        remove_unused_columns=config.get("remove_unused_columns", False),
        resume_from_checkpoint=config.get("resume_from_checkpoint", False),
        seed=42,
    )

# Main for testing
def main():
    # Load training configuration
    config = load_config("config.json")
    training_args = get_training_args(config)
    print(training_args)

    # Load model configuration
    model_config = "model_config.json"
    model_config = VivitConfig.from_json_file(model_config)
    print(model_config)

if __name__ == "__main__":
    main()
    