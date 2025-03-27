from src.utils.utils import load_dataset, get_image_processor, collate_fn
from src.models.model_utils import load_model
from src.trainers.trainer import LogTrainer
import os
import wandb
from src.models.model_testing import run_inference_and_save
from src.scripts.prediction_analysis import generate_predictions_report
import torch
from transformers import VivitConfig, HfArgumentParser, TrainerCallback
from src.trainers.TrainingArguments_projection import TrainingArguments_projection
import hydra
from omegaconf import DictConfig, OmegaConf
from src.config.hydra_config import update_experiment_name, write_configs_to_json

from accelerate import Accelerator


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Set device variable to use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        # Get the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f'Using GPU. Number of GPUs available: {num_gpus}')
        
        # List all available GPUs
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f'GPU {i}: {gpu_name}')
    else:
        print('CUDA is not available. Using CPU.')

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
    # Sanitize the project name by using only the substring after the last '/'
    project_name = cfg.trainer_config.dataset_folder.split('/')[-1]
    os.environ["WANDB_PROJECT"] = f"catdx_{project_name}"

    # Load model
    model = load_model(vivit_config=vivit_config, is_pretrained=True)
    model.to(device)
    print(model)
    
    # Training arguments
    parser = HfArgumentParser(TrainingArguments_projection)
    training_args, = parser.parse_json_file(json_file=trainer_json, allow_extra_keys=True)

    class EmptyCacheCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()
            step = state.global_step

    # Create Trainer
    trainer = LogTrainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator = lambda examples: collate_fn(examples, image_processor, cfg.model_config.num_channels),
        # callbacks=[EmptyCacheCallback()]
    )

    if cfg.trainer_config.gather_loss:
        accelerator = Accelerator()
        model, trainer = accelerator.prepare(model, trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)

    if cfg.trainer_config.gather_loss:
        # Save the model using accelerator.unwrap_model.
        accelerator.wait_for_everyone()  # Ensure all processes are synced.
        model = accelerator.unwrap_model(model)

    # Save the model
    model.save_pretrained(cfg.experiment_name)

    # Run inference and save results only if training mode contains regression
    if 'regression' in cfg.trainer_config.training_mode:
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
