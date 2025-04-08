import pandas as pd
import os
import torch

# Function to perform inference and collect predictions for each partition
def perform_inference(dataset, splits, trainer):
    """
    Performs inference on each available split in the dataset using the trainer.
    Returns a dictionary with raw prediction outputs per split.
    """
    raw_results = {}
    available_splits = dataset.keys()

    for split in splits:
        if split not in available_splits:
            print(f"Skipping {split} partition since it does not exist in the dataset.")
            continue
        print(f"Performing inference on {split} partition...")
        # predictions = trainer.predict(dataset[split])
        # raw_results[split] = predictions

        predictions = trainer.custom_predict_loop(dataset[split])
        raw_results[split] = predictions

        # Clear CUDA cache
        torch.cuda.empty_cache()

    return raw_results

def process_predictions(raw_results):
    """
    Processes raw prediction outputs from the inference function.
    Returns a dictionary with processed results for each split.
    Each result item is a dict with 'actual' and 'predicted' keys.
    """
    processed_results = {}
    for split, predictions in raw_results.items():
        actual_labels = predictions['labels']
        # If predictions contain a tuple, take the first item; otherwise use predictions directly
        predicted_labels = predictions['logits'][0] if isinstance(predictions['logits'], tuple) else predictions['logits']

        processed_results[split] = []
        for prediction, actual_label in zip(predicted_labels, actual_labels):
            label = actual_label.item() if isinstance(actual_label, torch.Tensor) else actual_label
            value = prediction[0]
            if isinstance(value, torch.Tensor):
                value = value.item()
            processed_results[split].append({
                'actual': label,
                'predicted': value,
            })
        print(f"Processing predictions for {split} partition completed.")

    return processed_results

# Function to save the results to a CSV file in the model's folder
def save_results(results, output_dir):
    saved_files = []

    # Save results for each partition separately
    for split, split_results in results.items():
        if split_results:  # Check if the results list is not empty
            print(f"Saving predictions for {split} partition...")
            output_file = os.path.join(output_dir, f'predictions_{split}.csv')
            split_results = [dict(zip(['actual', 'predicted'], item.values())) for item in split_results]
            df = pd.DataFrame(split_results)
            df.to_csv(output_file, index=False)
            saved_files.append(output_file)
            print(f"Predictions for {split} partition saved to {output_file}")
        else:
            print(f"No predictions to save for {split} partition.")

    return saved_files

# Function to load the saved predictions from a CSV file
def load_predictions(csv_file):
    return pd.read_csv(csv_file)

# Callable function to run the entire process and return paths of generated CSVs
def run_inference_and_save(dataset, trainer, output_dir, splits=['train', 'validation', 'test']):

    # Print available splits in the dataset
    print(f"Available splits in the dataset: {', '.join(dataset.keys())}")

    # Perform inference and collect predictions
    results = perform_inference(dataset, splits, trainer)
    
    # Process the raw results to extract actual and predicted labels
    results = process_predictions(results)

    # Save the results to a separate CSV file for each split
    saved_files = save_results(results, output_dir)

    # Return the list of CSV paths
    return saved_files
