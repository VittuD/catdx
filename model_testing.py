import torch
import pandas as pd
import os
from tqdm import tqdm

# Function to preprocess the video for inference
def preprocess_example(example, image_processor, num_frames=32):
    video = example['pixel_values']
    frames = [frame.asnumpy() for frame in video]
    if len(frames) < num_frames:
        frames = frames * (num_frames // len(frames)) + frames[:num_frames % len(frames)]
    else:
        frames = frames[:num_frames]
    processed_video = image_processor(frames, return_tensors='pt')
    return processed_video['pixel_values']

# Function to perform inference and collect predictions for each partition
def perform_inference(dataset, model, image_processor, splits):
    results = {split: [] for split in splits}  # Store results separately for each split

    # Get the available splits in the dataset
    available_splits = dataset.keys()

    # Iterate over the provided splits list
    for split in splits:
        if split not in available_splits:
            print(f"Skipping {split} partition since it does not exist in the dataset.")
            continue  # Skip this partition if it doesn't exist
        print(f"Processing {split} partition...")

        # Iterate over the split with a progress bar
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            # Preprocess the example
            pixel_values = preprocess_example(example, image_processor)

            # Perform inference
            with torch.no_grad():
                pixel_values = pixel_values.to('cuda')
                outputs = model(pixel_values)
                prediction = outputs['logits'].squeeze().item()  # For regression, logits will contain the output value

            # Get the actual label and append the result
            actual_label = example['labels']
            results[split].append({
                'actual': actual_label,
                'predicted': prediction
            })

    return results

# Function to save the results to a CSV file in the model's folder
def save_results(results, output_dir):
    saved_files = []

    # Save results for each partition separately
    for split, split_results in results.items():
        if split_results:  # Check if the results list is not empty
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
def run_inference_and_save(dataset, model, trainer, output_dir, image_processor, splits=['train', 'validation', 'test']):

    # Print available splits in the dataset
    print(f"Available splits in the dataset: {', '.join(dataset.keys())}")

    # Perform inference and collect predictions
    results = perform_inference(dataset, model, image_processor, splits)

    # Save the results to a separate CSV file for each split
    saved_files = save_results(results, output_dir)

    # Return the list of CSV paths
    return saved_files
