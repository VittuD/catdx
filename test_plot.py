import os
from prediction_analysis import generate_predictions_report
from model_testing import run_inference_and_save
from utils import get_image_processor, load_config, load_dataset
from transformers import VivitForVideoClassification

def main():

    # Load configuration
    config = load_config()

    output_dir = config["run_name"]
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset and image processor
    dataset = load_dataset(config["dataset_folder"])
    image_processor = get_image_processor(config["resize_to"])

    model_name = "apical4_none_420p_amp_wd_lr/checkpoint-1014"
    model = VivitForVideoClassification.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    model.eval().to('cuda')

    print(f'Loaded model from {model_name}')
    # Run inference and save results
    results = run_inference_and_save(dataset=dataset, model=model, trainer=None, output_dir=model_name, image_processor=image_processor)
    
    # Generate predictions report
    for result in results:
        generate_predictions_report(result)

    # Load configuration and model
    #model_paths = [
    #    "apical4_none_analysis_evaluate",
    #    "apical4_none_224p",
    #    "apical4_none_224p_amp_freeze",
    #    "apical4_none_420p",
    #    "apical4_none_420p_amp_freeze",
    #    "apical4_none_224p_amp",
    #    "apical4_none_224p_amp_wd",
    #    "apical4_none_420p_amp",
    #    "apical4_none_420p_amp_wd"
    #]
#
    #for model_path in model_paths:
    #    # Results are already saved in the folder as predictions_train.csv, predictions_val.csv and predictions_test.csv
    #    # Create a list concatenating the model path to the results path
    #    results = [
    #        os.path.join(model_path, "predictions_train.csv"),
    #        os.path.join(model_path, "predictions_validation.csv"),
    #        os.path.join(model_path, "predictions_test.csv")
    #    ]
#
    #    # Generate predictions report
    #    for result in results:
    #        generate_predictions_report(result)


if __name__ == "__main__":
    main()
