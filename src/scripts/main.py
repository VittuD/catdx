from src.utils.utils import load_dataset, get_image_processor, collate_fn, safe_cat
from src.models.model_utils import load_model
from src.trainers.trainer import LogTrainer
import os
import wandb
from src.models.model_testing import run_inference_and_save, save_results, process_predictions
from src.scripts.prediction_analysis import generate_predictions_report
import torch
from transformers import VivitConfig, HfArgumentParser, TrainerCallback, EarlyStoppingCallback
from src.trainers.TrainingArguments_projection import TrainingArguments_projection
import hydra
from omegaconf import DictConfig
from src.config.hydra_config import update_experiment_name, write_configs_to_json
from accelerate import Accelerator, DistributedDataParallelKwargs
from src.estimators.estimators import AgeEstimator
import torch.distributed as dist
from omegaconf import OmegaConf


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

@accelerator.on_main_process
def print_gpus(device):
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

@accelerator.on_main_process
def wandb_start(secrets_file):
    # Checks on the path and file
    if not os.path.exists(secrets_file):
        raise FileNotFoundError(f"Secrets file {secrets_file} not found.")
    if not os.path.isfile(secrets_file):
        raise ValueError(f"Expected a file, but found a directory: {secrets_file}")
    if not os.access(secrets_file, os.R_OK):
        raise PermissionError(f"Permission denied to read the secrets file: {secrets_file}")
    # Log to wandb, they key must be in the file .secrets
    key = open(secrets_file).read().strip()
    wandb.login(key=key)

@accelerator.on_main_process
def run_inference_directly(cfg, dataset, trainer):
    results = run_inference_and_save(dataset=dataset, trainer=trainer, output_dir=cfg.experiment_name)

    # Generate predictions report
    pdf_files = []
    for result in results:
        pdf_files.append(generate_predictions_report(result))
    # Log each PDF file to wandb as its own artifact with a dynamically extracted alias
    for pdf_file in pdf_files:
        wandb.save(pdf_file)

@accelerator.on_main_process
# TODO this might be a duplicate at the moment, check if it can be removed
def run_inference_regressor(cfg, dataset, model, trainer):
    # Use the model's forward to get raw outputs (before the regressor) with torch.no_grad
    raw_outputs = {}
    for partition_name, partition_data in dataset.items():
        raw_outputs[partition_name] = trainer.predict(partition_data)

    # Process the raw outputs to get the predictions
    processed_results = process_predictions(raw_outputs)

    # For each raw output partition, save the predictions to a CSV file
    saved_files = save_results(processed_results, cfg.experiment_name)

    # Generate predictions report
    pdf_files = []
    for file in saved_files:
        pdf_files.append(generate_predictions_report(file))
    # Log each PDF file to wandb as its own artifact with a dynamically extracted alias
    for pdf_file in pdf_files:
        wandb.save(pdf_file)

    return raw_outputs

def custom_predict_loop(trainer, dataset):
    predictions = trainer.custom_predict_loop(dataset)
    # Predictions has structure:
    # {
    #     'train': {'loss': ..., 'logits': ..., 'projections': ..., 'hidden_states': ..., 'labels': ...},
    #     'validation': {'loss': ..., 'logits': ..., 'projections': ..., 'hidden_states': ..., 'labels': ...},
    #     'test': {'loss': ..., 'logits': ..., 'projections': ..., 'hidden_states': ..., 'labels': ...},
    # }
    # Each entry in the list is a dictionary with keys 'loss', 'logits', 'projections', 'hidden_states', and 'labels'
    if dist.is_initialized():
        # Gather predictions from all processes using all_gather_object
        gathered_predictions = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_predictions, predictions)
        # Concatenate the gathered predictions
        complete_predictions = {}
        # gathered_predictions is a list on n_machine elements, iterate over it
        for i in range(len(gathered_predictions)):
            for split_name in gathered_predictions[i].keys():
                current = gathered_predictions[i][split_name]
                existing = complete_predictions.get(split_name, {})

                complete_predictions[split_name] = {
                    'loss': safe_cat(existing.get('loss'), current['loss']),
                    'logits': safe_cat(existing.get('logits'), current['logits']),
                    'projections': safe_cat(existing.get('projections'), current['projections']),
                    'hidden_states': safe_cat(existing.get('hidden_states'), current['hidden_states']),
                    'labels': safe_cat(existing.get('labels'), current['labels']),
                }

    else:
        # If not distributed, just use the predictions as is
        complete_predictions = predictions
    return complete_predictions

@accelerator.on_main_process
def compute_mae_on_partitions(data):
    estimator = AgeEstimator()

    print("Training estimator")
    # data['train'] is a list of dicts where each dict contains keys like 'loss', 'logits', 'projections', 'hidden_states', and 'labels'
    mae_results = {}
    for split_name, split_data in data.items():
        # split_data is a list of dicts with keys like 'hidden_states' and 'labels'
        labels = split_data['labels'].cpu().numpy()

        cls_tokens = split_data['hidden_states'].cpu().numpy()

        if split_name == 'train':
            mae = estimator.fit(cls_tokens, labels)
        else:
            mae = estimator.score(cls_tokens, labels)
        mae_results[split_name] = mae

    return mae_results


@accelerator.on_main_process
def wandb_init_for_regression(original_name, cfg):
    wandb.finish()

    regression_name = f"{original_name}_regression"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "default_project"),
        name=regression_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

def train_regression_head(cfg, dataset, model):
    # Create a new trainer
    cfg.trainer_config.training_mode = 'regression'
    cfg.trainer_config.auto_find_batch_size = True
    cfg.trainer_config.num_train_epochs = 500
    cfg.trainer_config.learning_rate = 5e-3
    cfg.trainer_config.weight_decay = 0
    cfg.trainer_config.is_unsupervised = False

    parser = HfArgumentParser(TrainingArguments_projection)
    training_args, = parser.parse_dict(cfg.trainer_config, allow_extra_keys=True)
    
    new_trainer = LogTrainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        data_collator=lambda examples: collate_fn(
            examples,
            get_image_processor(model.config.image_size, cfg.model_config.num_channels),
            cfg.model_config.num_channels
        ),
    )

    wandb_init_for_regression(cfg.experiment_name, cfg)

    # Create a new 'regression' folder and change dir into that
    regression_dir = os.path.join(cfg.experiment_name, 'regression')
    os.makedirs(regression_dir, exist_ok=True)
    os.chdir(regression_dir)

    new_trainer.train()
    
    model.save_pretrained(cfg.experiment_name)

    os.chdir(cfg.experiment_name)

    # Perform inference on the validation set
    # Theoretically, this should be catched by the main loop since we changed the training mode
    # run_inference_directly(cfg, dataset, new_trainer)5

@accelerator.on_main_process
def save_results_pdf(results, output_dir):
    saved_files = save_results(results, output_dir)
    # Generate predictions report
    pdf_files = []
    for file in saved_files:
        pdf_files.append(generate_predictions_report(file))
    # Log each PDF file to wandb as its own artifact with a dynamically extracted alias
    for pdf_file in pdf_files:
        wandb.save(pdf_file)


class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.trainer.plot = "train"
    
    # TODO This triggers after the loss forward so it does not log anything. MAybe on_epoch_end (?)
    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.plot = "eval"




@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Set device variable to use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print the available GPUs
    print_gpus(device)

    # Update the run name if necessary.
    base_dir = os.getcwd()
    cfg = update_experiment_name(cfg, base_dir=base_dir)
    
    # Use the updated run name (from trainer_config) as the experiment directory.
    experiment_dir = cfg.experiment_name
    os.makedirs(experiment_dir, exist_ok=True)

    # Log to wandb, they key must be in the file .secrets
    wandb_start(cfg.secrets_file)

    # Set wandb project
    project_name = cfg.trainer_config.dataset_folder.split('/')[-1]
    os.environ["WANDB_PROJECT"] = f"catdx_{project_name}"

    # Set working directory to the home dir for dataset and secrets
    os.chdir('/scratch/catdx/')

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

    # Load model
    model = load_model(vivit_config=vivit_config, is_pretrained=True)
    model.to(device)
    
    # Training arguments
    parser = HfArgumentParser(TrainingArguments_projection)
    training_args, = parser.parse_json_file(json_file=trainer_json, allow_extra_keys=True)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=15,
        early_stopping_threshold=0.0
    )

    # Create Trainer
    trainer = LogTrainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator = lambda examples: collate_fn(examples, image_processor, cfg.model_config.num_channels),
    )

    trainer.add_callback(MyCallback(trainer))
    trainer.add_callback(early_stopping_callback)

    if cfg.trainer_config.gather_loss:
        model, trainer = accelerator.prepare(model, trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)

    if cfg.trainer_config.gather_loss:
        # Save the model using accelerator.unwrap_model.
        accelerator.wait_for_everyone()  # Ensure all processes are synced.
        model = accelerator.unwrap_model(model)

    # Save the model
    model.save_pretrained(cfg.experiment_name)

    # Run sci-kit regressor only if training mode contains contrastive
    if 'contrastive' in cfg.trainer_config.training_mode:# Train a regressor (estimators.py Ridge regression) on every class token of the 'train' partition
        # data = custom_predict_loop(trainer, dataset)
        # processed_data = compute_mae_on_partitions(data)
        # print(processed_data)
        # # Log processed data for each split to wandb
        # for split_name, mae in processed_data.items():
        #     wandb.log({f"{split_name}_mae_ridge": mae})
        
        # Use train_regression_head
        train_regression_head(cfg, dataset, model)
        
    
    # Run inference and save results only if training mode contains regression
    data = custom_predict_loop(trainer, dataset)
    processed_data = process_predictions(data)
    os.chdir('..')
    save_results_pdf(processed_data, cfg.experiment_name)

if __name__ == '__main__':
    main()
