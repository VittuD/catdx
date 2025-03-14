import decord
import os

# function to validae the video using decord library given a path
def is_valid(video):
    try:
        decord.VideoReader(video)
        return True
    except Exception:
        return False
    
# for each .mp4 file in a folder print if it is valid or not
def validate_videos(folder):
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            # print(f"Checking {file}")
            if not is_valid(os.path.join(folder, file)):
                print(f"Invalid video: {file}")
            # else:
                # print(f"Valid video: {file}")

dataset_train_path = "dataset_apical4_none/train"
dataset_val_path = "dataset_apical4_none/val"
print("Checking train dataset")
validate_videos(dataset_train_path)
print("Checking validation dataset")
validate_videos(dataset_val_path)
