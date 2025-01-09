import os
from prediction_analysis import generate_predictions_report
from model_testing import run_inference_and_save
from utils import get_image_processor, load_dataset
from trainer import LogTrainer
from config import get_training_args, load_config
from utils import compute_metrics, collate_fn
from model import VivitWithOptionalProjectionHead
from transformers import VivitConfig

def main():

    # Load configuration
    config = load_config()
    model_config = "model_config.json"

    output_dir = config["run_name"]
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset and image processor
    dataset = load_dataset(config["dataset_folder"])
    image_processor = get_image_processor(config["resize_to"])

    model_name = "apical4_none_vanilla_regression_debug/checkpoint-51"
    vivit_config = VivitConfig.from_json_file(model_config)

    # model = VivitForVideoClassification.from_pretrained(
    #     pretrained_model_name_or_path=model_name
    # )
    
    model = VivitWithOptionalProjectionHead.from_pretrained(pretrained_model_name_or_path = model_name, config=vivit_config)
    model.to('cuda')
    print(f'Loaded model from {model_name} with config {vivit_config}')

    # Load trainer for predictions
    training_args = get_training_args(config)
    # Create Trainer
    trainer = LogTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator = lambda examples: collate_fn(examples, image_processor),
        compute_metrics=compute_metrics,
    )

    # Run inference and save results
    results = run_inference_and_save(dataset=dataset, trainer=trainer, output_dir=model_name)
    print(results)
    # Generate predictions report
    for result in results:
        print(f'Processing result: {result}')
        generate_predictions_report(result)

if __name__ == "__main__":
    main()
