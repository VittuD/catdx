from utils import load_dataset, get_image_processor, collate_fn
from model_utils import load_model
from config import load_config, get_training_args
from trainer import LogTrainer
import os
import wandb
from model_testing import run_inference_and_save
from prediction_analysis import generate_predictions_report
import torch
from transformers import VivitConfig, HfArgumentParser
from arg_parser import parse_args, update_config
from TrainingArguments_projection import TrainingArguments_projection
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra_config import update_experiment_name, write_configs_to_json

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
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

    # Update the run name if necessary.
    base_dir = os.getcwd()
    cfg = update_experiment_name(cfg, base_dir=base_dir)
    
    # Use the updated run name (from trainer_config) as the experiment directory.
    experiment_dir = cfg.experiment_name
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set working directory to the home dir for dataset and secrets
    os.chdir('/scratch/catdx/')

    # Log to wandb, they key must be in the file .secrets
    key = open('.secrets').read().strip()
    wandb.login(key=key)

    # Load dataset and image processor
    transforms = None
    dataset = load_dataset(cfg.trainer_config.dataset_folder, augmentation=transforms)

    # Go back to the base dir
    os.chdir(base_dir)
    
    # Write out the model and trainer configs as JSON files.
    model_json, trainer_json = write_configs_to_json(cfg, experiment_dir)
    
    # Load Vivit configuration
    vivit_config = VivitConfig.from_json_file(model_json)

    image_processor = get_image_processor(vivit_config.image_size, cfg.model_config.num_channels)
    
    # Set wandb project
    os.environ["WANDB_PROJECT"] = f"catdx_{cfg.trainer_config.dataset_folder}"

    # Load model
    model = load_model(vivit_config=vivit_config, is_pretrained=True)
    model.to('cuda')

    # Training arguments
    parser = HfArgumentParser(TrainingArguments_projection)
    training_args, = parser.parse_json_file(json_file=trainer_json, allow_extra_keys=True)

    # Create Trainer
    trainer = LogTrainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator = lambda examples: collate_fn(examples, image_processor, cfg.model_config.num_channels),
    )

    # Train the model
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)

    # Save the model
    model.save_pretrained(cfg.experiment_name)

    # Run inference and save results
    results = run_inference_and_save(dataset=dataset, trainer=trainer, output_dir=cfg.experiment_name)

    # Generate predictions report
    pdf_files = []
    for result in results:
        pdf_files.append(generate_predictions_report(result))

    # Log each PDF file to wandb as its own artifact with a dynamically extracted alias
    for pdf_file in pdf_files:
        wandb.save(pdf_file)


if __name__ == '__main__':
    main()
