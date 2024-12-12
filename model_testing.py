from transformers import VivitForVideoClassification, VivitImageProcessor
import datasets
import torch
import pandas as pd
import os
from tqdm import tqdm

# Function to load the trained model
def load_model(model_path):
    model = VivitForVideoClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to load the image processor
def load_image_processor(resize_to=224):
    image_processor = VivitImageProcessor(
        do_resize=True,
        size={'height': resize_to, 'width': resize_to},
        do_center_crop=False,  # Disable cropping
        do_normalize=True,
    )
    return image_processor

# Function to load the dataset
def load_dataset(dataset_path):
    dataset = datasets.load_dataset(dataset_path)
    return dataset

# Function to preprocess the video for inference
def preprocess_example(example, image_processor, num_frames=32):
    video = example['video']
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
                prediction = outputs.logits.squeeze().item()  # For regression, logits will contain the output value
            
            # Get the actual label and append the result
            actual_label = example['CO']
            results[split].append({
                'actual': actual_label,
                'predicted': prediction
            })
    
    return results

# Function to save the results to a CSV file in the model's folder
def save_results(results, model_path):
    model_dir = os.path.dirname(model_path)
    saved_files = []

    # Save results for each partition separately
    for split, split_results in results.items():
        output_file = os.path.join(model_dir, f'predictions_{split}.csv')
        df = pd.DataFrame(split_results)
        df.to_csv(output_file, index=False)
        saved_files.append(output_file)
        print(f"Predictions for {split} partition saved to {output_file}")
    
    return saved_files

# Callable function to run the entire process and return paths of generated CSVs
def run_inference_and_save(model_path, dataset_path, splits=['train', 'validation', 'test']):
    # Load model, image processor, and dataset
    model = load_model(model_path)
    image_processor = load_image_processor(resize_to=224)
    dataset = load_dataset(dataset_path)
    model.to('cuda')  # Move the model to GPU

    # Print available splits in the dataset
    print(f"Available splits in the dataset: {', '.join(dataset.keys())}")
    
    # Perform inference and collect predictions
    results = perform_inference(dataset, model, image_processor, splits)
    
    # Save the results to a separate CSV file for each split
    saved_files = save_results(results, model_path)
    
    # Return the list of CSV paths
    return saved_files
