from utils import load_dataset, get_image_processor, collate_fn
from model_utils import load_model
from config import load_config, get_training_args
from trainer import LogTrainer
import os
import wandb
from model_testing import run_inference_and_save
from prediction_analysis import generate_predictions_report
import torch
from transformers import VivitConfig
from arg_parser import parse_args, update_config

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f'Number of GPUs available: {num_gpus}')

        # List all available GPUs
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f'GPU {i}: {gpu_name}')
    else:
        print('CUDA is not available. No GPUs detected.')

    # Load Vivit configuration
    vivit_config = VivitConfig.from_json_file('model_config.json')

    # Load configuration
    config = load_config()

    # Load CLI arguments
    args = parse_args()
    config = update_config(config, args)

    output_dir = config['run_name']
    os.makedirs(output_dir, exist_ok=True)

    # Log to wandb, they key must be in the file .secrets
    key = open('.secrets').read().strip()
    wandb.login(key=key)

    # Load model
    model = load_model(vivit_config=vivit_config, is_pretrained=True)
    model.to('cuda')
    # model = load_model(vivit_config=vivit_config)

    # Load dataset and image processor
    dataset = load_dataset(config['dataset_folder'])
    image_processor = get_image_processor(vivit_config.image_size)

    # Training arguments
    training_args = get_training_args(config)

    # Create Trainer
    trainer = LogTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator = lambda examples: collate_fn(examples, image_processor),
        # compute_metrics=compute_metrics,
        training_mode=vivit_config.training_mode,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)

    # Save the model
    model.save_pretrained(config['run_name'])

    # Run inference and save results
    results = run_inference_and_save(dataset=dataset, trainer=trainer, output_dir=config['run_name'])

    # Generate predictions report
    for result in results:
        generate_predictions_report(result)

## TODO Add the option to pass arguments via command line (they should override the config file)
if __name__ == '__main__':
    main()
