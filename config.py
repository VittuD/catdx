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
        logging_first_step=config.get("logging_first_step", False),
        logging_strategy=config.get("save_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        fp16=config.get("fp16", False),
        remove_unused_columns=config.get("remove_unused_columns", False),
        seed=42,
    )

def get_vivit_config(num_frames, resize_to, config, model_name):
    vivit_config = VivitConfig.from_pretrained(model_name)
    vivit_config.num_frames = num_frames
    vivit_config.num_labels = 1  # For regression
    vivit_config.problem_type = 'regression'
    vivit_config.video_size = [num_frames, resize_to, resize_to]
    vivit_config.image_size = resize_to
    vivit_config.tubelet_size=config.get("tubelet_size", "[2, 16, 16]")
    vivit_config.freeze = config.get("freeze", [])
    vivit_config.vivit_training_mode = config.get("vivit_training_mode", "")
    return vivit_config

def get_vivit_config_from_json(config_json, model_name):
    #vivit_config = VivitConfig.from_pretrained(model_name)
    vivit_config = VivitConfig.from_json_file(config_json)
    return vivit_config

# Main for testing
def main():
    # Load configuration
    config = load_config("config.json")
    model_config = "model_config.json"
    model_config = get_vivit_config_from_json(model_config, "google/vivit-b-16x2")
    print(model_config)

if __name__ == "__main__":
    main()
    