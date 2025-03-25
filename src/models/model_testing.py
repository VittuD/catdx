import pandas as pd
import os
import torch

# Function to perform inference and collect predictions for each partition
def perform_inference(dataset, splits, trainer):
    results = {split: [] for split in splits}  # Store results separately for each split

    # Get the available splits in the dataset
    available_splits = dataset.keys()

    # Iterate over the provided splits list
    for split in splits:
        if split not in available_splits:
            print(f"Skipping {split} partition since it does not exist in the dataset.")
            continue  # Skip this partition if it doesn't exist
        print(f"Processing {split} partition...")
        
        # Use the trainer predict method to get predictions on the split
        # TODO use trainer evaluate to avoid OOM (not keeping gradients)
        predictions = trainer.predict(dataset[split])
        actual_labels = predictions.label_ids
        predicted_labels = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions

        # Iterate over the predictions and actual labels
        for prediction, actual_label in zip(predicted_labels, actual_labels):
            results[split].append({
            'actual': actual_label,
            'predicted': prediction[0]  # Extract the number directly from the prediction array
            })

        print(f"Predictions for {split} partition completed.")

        # Clear cuda cache
        torch.cuda.empty_cache()

    return results

# Function to save the results to a CSV file in the model's folder
def save_results(results, output_dir):
    saved_files = []

    # Save results for each partition separately
    for split, split_results in results.items():
        if split_results:  # Check if the results list is not empty
            print(f"Saving predictions for {split} partition...")
            output_file = os.path.join(output_dir, f'predictions_{split}.csv')
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

    # Save the results to a separate CSV file for each split
    saved_files = save_results(results, output_dir)

    # Return the list of CSV paths
    return saved_files
