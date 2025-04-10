import os
import random
import pandas as pd

def val_csv(train_csv_path, val_csv_path, val_split=0.2):
    """
    Create a validation split from the training dataset.

    Args:
        train_csv_path (str): Path to the training CSV file.
        val_csv_path (str): Path to save the validation CSV file.
        val_split (float|int): If float, interpreted as the percentage of videos (0-1).
                              If int, interpreted as the number of videos.

    Returns:
        val_df (pd.DataFrame): DataFrame containing validation set.
        new_train_df (pd.DataFrame): DataFrame containing updated training set.
    """
    # Load the training CSV
    train_df = pd.read_csv(train_csv_path)

    # Get the unique videos
    unique_videos = train_df['file_name'].unique().tolist()
    num_videos = len(unique_videos)

    # Determine the number of videos for validation set
    if isinstance(val_split, float):
        if 0 < val_split < 1:
            num_val_videos = int(num_videos * val_split)
        else:
            raise ValueError("val_split as a float should be between 0 and 1.")
    elif isinstance(val_split, int):
        if 0 < val_split <= num_videos:
            num_val_videos = val_split
        else:
            raise ValueError("val_split as an int should be between 1 and the total number of videos.")
    else:
        raise TypeError("val_split should be a float (0-1) or an int (number of videos).")

    # Get a random sample of videos for the validation set
    val_videos = random.sample(unique_videos, num_val_videos)

    # Create the validation set
    val_df = train_df[train_df['file_name'].isin(val_videos)]

    # Create the new training set by excluding the validation videos
    new_train_df = train_df[~train_df['file_name'].isin(val_videos)]

    # Save the validation set to a new CSV file
    val_df.to_csv(val_csv_path, index=False)

    # Save the new training set to the original CSV file
    new_train_df.to_csv(train_csv_path, index=False)

    return val_df, new_train_df

def move_videos(video_dir, val_video_dir, val_df):
    """
    Move videos to the validation directory.

    Args:
        video_dir (str): Directory containing the videos.
        val_video_dir (str): Directory to save the validation videos.
        val_df (pd.DataFrame): DataFrame containing validation set information.

    Returns:
        val_video_dir (str): Path to the validation video directory.
    """
    # Create the validation video directory if it doesn't exist
    os.makedirs(val_video_dir, exist_ok=True)

    # Move the videos to the validation directory
    for video_file in val_df['file_name'].unique():
        video_path = os.path.join(video_dir, video_file)
        val_video_path = os.path.join(val_video_dir, video_file)
        if os.path.exists(video_path):
            os.rename(video_path, val_video_path)
        else:
            print(f"Warning: {video_path} does not exist and cannot be moved.")

    return val_video_dir

# Define the paths
train_csv_path = 'dataset_apical4_none/train/metadata.csv'
val_csv_path = 'dataset_apical4_none/val/metadata.csv'
video_dir = 'dataset_apical4_none/train/'
val_video_dir = 'dataset_apical4_none/val/'

# Create the validation split (update val_split as needed)
val_split = 69  # Can be a float (percentage) or int (number of videos)
val_df, new_train_df = val_csv(train_csv_path, val_csv_path, val_split)

# Move the videos to the validation directory
val_video_dir = move_videos(video_dir, val_video_dir, val_df)

# Print the paths of the new CSV files and the validation video directory
print(f"Validation CSV file saved to: {val_csv_path}")
