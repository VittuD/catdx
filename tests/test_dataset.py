import os
import pandas as pd
from datasets import Dataset, Features, Value, Video, DatasetDict
import datasets

# Define the folder containing your .mp4 files
video_folder = "fix_apical4"
csv_file = "fix_apical4/all_files_with_partition.csv"

test_dataset = datasets.load_dataset(video_folder)
# Load the combined CSV that contains: file_name, CO, partition
df = pd.read_csv(csv_file)

features = Features({
    "file_name": Value("string"),
    "CO": Value("float"),
    "partition": Value("string"),
})

dataset = Dataset.from_pandas(df, features=features)
# Extract only the video column from test dataset
video_column = test_dataset["train"].select_columns(["video"])

dataset = dataset.add_column("video", test_dataset["train"].select_columns(["video"]))
train_dataset = dataset.filter(lambda x: x["partition"] == "train")
val_dataset = dataset.filter(lambda x: x["partition"] == "val")
test_dataset = dataset.filter(lambda x: x["partition"] == "test")
        
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset,
})

print(dataset)
