from transformers import TrainingArguments, VivitConfig

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
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=config["eval_steps"] * 2,
        save_steps=config["eval_steps"] * 2,
        logging_steps=config["eval_steps"] / 5,
        load_best_model_at_end=True,
        seed=42,
    )

def get_vivit_config(num_frames, resize_to, config, model_name):
    vivit_config = VivitConfig.from_pretrained(model_name)
    vivit_config.num_frames = num_frames
    vivit_config.num_labels = 1  # For regression
    vivit_config.problem_type = 'regression'
    vivit_config.video_size = [num_frames, resize_to, resize_to]
    return vivit_config
