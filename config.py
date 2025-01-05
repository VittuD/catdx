from transformers import TrainingArguments, VivitConfig
import json

# Load configuration from a JSON file
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)
    
def get_training_args(config):
    return TrainingArguments(
        output_dir=config["run_name"],
        report_to=config["report_to"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        logging_first_step= config["logging_first_step"],
        logging_strategy= config["save_strategy"],
        save_strategy= config["save_strategy"],
        eval_strategy= config["eval_strategy"],
        fp16=config["fp16"],
        remove_unused_columns = config["remove_unused_columns"],
        seed=42,
    )

def get_vivit_config(num_frames, resize_to, config, model_name):
    vivit_config = VivitConfig.from_pretrained(model_name)
    vivit_config.num_frames = num_frames
    vivit_config.num_labels = 1  # For regression
    vivit_config.problem_type = 'regression'
    vivit_config.video_size = [num_frames, resize_to, resize_to]
    vivit_config.image_size = resize_to
    vivit_config.tubelet_size=eval(config["tubelet_size"])
    vivit_config.freeze = config["freeze"]
    return vivit_config
