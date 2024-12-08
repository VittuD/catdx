from transformers import VivitForVideoClassification, VivitImageProcessor
import datasets
import torch
import pandas as pd
from tqdm import tqdm

# Load the trained model
model = VivitForVideoClassification.from_pretrained('vivit_apical4_none_pretrained_mse_loss/checkpoint-3450')
model.eval()  # Set the model to evaluation mode

# Load the image processor
resize_to = 224
image_processor = VivitImageProcessor(
    do_resize=True,
    size={'height': resize_to, 'width': resize_to},
    do_center_crop=False,  # Disable cropping
    do_normalize=True, 
)

# Load the dataset
dataset = datasets.load_dataset('dataset_apical4_none')
model.to('cuda')

# Function to preprocess the video for inference
def preprocess_example(example):
    """
    Preprocess a single video example for inference.
    """
    video = example['video']  # Get the video object
    # Extract frames from the video
    frames = [frame.asnumpy() for frame in video]  # Convert Decord frames to numpy arrays
    # Preprocess the video frames
    processed_video = image_processor(frames, return_tensors="pt")  # Returns a dictionary
    return processed_video["pixel_values"]  # Extract pixel values

# List to store results
results = []

# Iterate over the validation split with a progress bar
for example in tqdm(dataset['validation'], desc="Processing examples"):
    # Preprocess the example
    pixel_values = preprocess_example(example)
    
    # Perform inference
    with torch.no_grad():
        pixel_values = pixel_values.to('cuda')
        outputs = model(pixel_values)
        prediction = outputs.logits.squeeze().item()  # For regression, logits will contain the output value
    
    # Get the actual label and file name
    actual_label = example['CO']
    
    # Append the result to the list
    results.append({
        'actual': actual_label,
        'predicted': prediction
    })

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
