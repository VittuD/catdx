## TODO broken as it is. Only outputs -1 to 1 values I think

import os
from prediction_analysis import generate_predictions_report
from model_testing import run_inference_and_save
from utils import get_image_processor, load_dataset
from transformers import VivitForVideoClassification
from trainer import LogTrainer
from config import get_training_args, load_config, get_vivit_config
from utils import compute_metrics, collate_fn
from model import load_model

def main():

    # Load configuration
    config = load_config()

    output_dir = config["run_name"]
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset and image processor
    dataset = load_dataset(config["dataset_folder"])
    image_processor = get_image_processor(config["resize_to"])

    model_name = "apical4_none_basic_regression_debug_lr/checkpoint-91"
    # model = VivitForVideoClassification.from_pretrained(
    #     pretrained_model_name_or_path=model_name
    # )
    num_frames = 32
    vivit_config = get_vivit_config(num_frames, config['resize_to'], config, model_name)
    model = load_model(vivit_config, model_name)
    model.eval().to('cuda')
    print(f'Loaded model from {model_name}')

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
