import numpy as np
import os
import pandas as pd
from datasets import Dataset, Features, Value, Video, Array4D, DatasetDict

dataset_path = '/shared/RVENetCropRsz'
dataset_csv = os.path.join(dataset_path, 'codebook_vivit.csv')
test_path = '/shared/RVENetCropRsz/train/train_1/0a65df5ec44d995714744a44c6357d5e_0000_0000-0.npy'
if os.path.exists(test_path):
    print("File exists")
else:
    print("File does not exist")

# Load the numpy array
data = np.load(test_path)
# print("Data shape:", data.shape)
# print("Data type:", data.dtype)
# 
# # Check the contents of the numpy array
# print("Data contents:", data)

## Dataset Utilities
def create_datasets(csv_file, video_folder):
    # 1) read CSV + resolve absolute paths
    df = pd.read_csv(csv_file)
    df["path"] = df["file_name"].apply(lambda x: os.path.abspath(os.path.join(video_folder, x)))
    df = df.drop(columns=["file_name"])

    # 2) define features: path stays as string, CO & partition as before
    features = Features({
        "path": Value("string"),
        "CO":   Value("float32"),
        "partition": Value("string"),
    })

    # 3) build the HF Dataset
    ds = Dataset.from_pandas(df, features=features)

    # 4) split
    ds_train = ds.filter(lambda x: x["partition"] == "train")
    ds_val   = ds.filter(lambda x: x["partition"] == "validation")
    ds_test  = ds.filter(lambda x: x["partition"] == "test")

    # 5) load the actual arrays into a new column “video” (batching=False so each example is separate)
    def _load_video(example):
        arr = np.load(example["path"])
        return {"video": arr}

    array_feature = Features({
        "video": Array4D(dtype="float32", shape=(None, 20, 224, 224)),
    })

    ds_train = ds_train.map(_load_video, remove_columns=["path"], features=array_feature, batched=False)
    ds_val   = ds_val.map(_load_video,   remove_columns=["path"], features=array_feature, batched=False)
    ds_test  = ds_test.map(_load_video,  remove_columns=["path"], features=array_feature, batched=False)

    return DatasetDict({
        "train":      ds_train,
        "validation": ds_val,
        "test":       ds_test,
    })

dataset = create_datasets(dataset_csv, dataset_path)
print(dataset)
