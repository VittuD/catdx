import os
import sys
import argparse
import json

from src.utils.utils import load_dataset, get_image_processor, collate_fn, compute_metrics
from src.models.model_utils import VivitWithOptionalProjectionHead
from src.trainers.trainer import LogTrainer
from src.models.model_testing import run_inference_and_save
from src.scripts.prediction_analysis import generate_predictions_report
from transformers import VivitConfig, HfArgumentParser
from src.trainers.TrainingArguments_projection import TrainingArguments_projection

def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions report from a checkpoint using run JSON configs."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory (e.g. .../test_e2e_reg/checkpoint-266)"
    )
    args = parser.parse_args()

    # Set wandb project env var to "report_runs" to avoid mixing with training runs.
    os.environ["WANDB_PROJECT"] = "report_runs"

    # Absolute path to the checkpoint directory.
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    # The run directory is assumed to be the parent of the checkpoint directory.
    run_dir = os.path.dirname(checkpoint_dir)
    
    # Load JSON configs from the run directory.
    trainer_config_file = os.path.join(run_dir, "trainer_config.json")
    model_config_file   = os.path.join(run_dir, "model_config.json")
    config_file         = os.path.join(run_dir, "config.json")
    
    for fname in [trainer_config_file, model_config_file, config_file]:
        if not os.path.exists(fname):
            print(f"Error: Expected file {fname} not found.")
            sys.exit(1)
    
    with open(trainer_config_file, "r") as f:
        trainer_config = json.load(f)
    with open(config_file, "r") as f:
        run_config = json.load(f)
    
    # Use the checkpoint directory for output folder.
    output_dir = checkpoint_dir
    
    # Load dataset and image processor.
    dataset = load_dataset(trainer_config["dataset_folder"])
    image_processor = get_image_processor(trainer_config["resize_to"], run_config["num_channels"])
    
    # Load model configuration.
    vivit_config = VivitConfig.from_json_file(model_config_file)
    
    # Load model from checkpoint.
    model = VivitWithOptionalProjectionHead.from_pretrained(
        checkpoint_dir,
        config=vivit_config,
        ignore_mismatched_sizes=True
    )
    model.to("cuda")
    print(model)
    # Print named parameters
    for name, param in model.named_parameters():
        print(name, param.size())
    print(f"Loaded model from checkpoint: {checkpoint_dir}")
    
    # Training arguments: using HfArgumentParser to load from trainer_config_file.
    parser = HfArgumentParser(TrainingArguments_projection)
    training_args, = parser.parse_json_file(json_file=trainer_config_file, allow_extra_keys=True)
    
    # Create the trainer.
    trainer = LogTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=lambda examples: collate_fn(examples, image_processor, run_config["num_channels"]),
        # compute_metrics=compute_metrics,
    )
    
    # Run inference and save predictions.
    try:
        results = run_inference_and_save(dataset=dataset, trainer=trainer, output_dir=output_dir)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
    
    print("Prediction CSV files:")
    for csv_path in results:
        print(f" - {csv_path}")
    
    # Generate prediction reports (PDFs) for each CSV.
    for csv_path in results:
        try:
            pdf_file = generate_predictions_report(csv_path)
            print(f"Generated report: {pdf_file}")
        except Exception as e:
            print(f"Error generating report for {csv_path}: {e}")

if __name__ == "__main__":
    main()
