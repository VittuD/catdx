from utils import load_config, load_dataset, get_image_processor, collate_fn, compute_metrics
from model import load_model
from config import get_training_args, get_vivit_config
from trainer import LogTrainer

def main():
    # Load configuration
    config = load_config()

    # Load dataset and image processor
    dataset = load_dataset(config["dataset_folder"])
    image_processor = get_image_processor(config["resize_to"])

    # Load configuration and model
    num_frames = len(dataset['train'][0]['pixel_values'])
    model_name = "google/vivit-b-16x2-kinetics400"
    vivit_config = get_vivit_config(num_frames, config["resize_to"], config, model_name)
    model = load_model(vivit_config, model_name)

    # Training arguments
    training_args = get_training_args(config)

    # Create Trainer
    trainer = LogTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=lambda examples: collate_fn(examples, image_processor),
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(config["run_name"])

if __name__ == "__main__":
    main()
