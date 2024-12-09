import pandas as pd
import numpy as np
import cv2
import os
import torch

apical4_path = 'data/apical4'
annotations_path = 'catdx/easy/existingLabel.xlsx'

# Load annotations
annotations = pd.read_excel(annotations_path)

# Create a csv with only patient and CO
annotations = annotations[['patient', 'CO']]
annotations.to_csv('annotations_CO.csv', index=False)

def convert_pt_to_mp4(pt_path, pt_filename, output_filename, fps=30):
    # Open a pt file with torch
    pt_file = os.path.join(pt_path, pt_filename)
    pt_video = torch.load(pt_file, weights_only=False)
    print(pt_video.shape)

    # Get the frame size from the video shape
    frame_size = (pt_video.shape[2], pt_video.shape[1])

    # Convert it to a video and save it as mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    for frame in pt_video:
        frame_np = frame.numpy()  # Convert torch.Tensor to numpy array
        frame_np = (frame_np * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        # frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
        out.write(frame_np)

    out.release()
    print('Video saved as ', output_filename)

# For each file present in the apical4_none folder, convert it to mp4 and store it in dataset_apical4_none folder
for file in os.listdir('catdx/easy/apical4'):
    if file.endswith('.pt'):
        output_filename = f'dataset_apical4_none/{file[:-3]}.mp4'
        if not os.path.exists(output_filename):
            convert_pt_to_mp4('catdx/easy/apical4', file, output_filename)

# Create the train and validation folders
os.makedirs('dataset_apical4_none/train', exist_ok=True)
os.makedirs('dataset_apical4_none/val', exist_ok=True)

# Read the train and test files from the txt file
partition_file = '/workspaces/catdx/catdx/train_test_split.txt'
with open(partition_file, 'r') as f:
    lines = f.read().splitlines()

train_files = []
val_files = []
is_train = True

for line in lines:
    if line == 'TRAIN':
        is_train = True
    elif line == 'TEST':
        is_train = False
    elif line:
        if is_train:
            train_files.append(line + '.mp4')
        else:
            val_files.append(line + '.mp4')

# Move the files
for file in train_files:
    train_path = os.path.join('dataset_apical4_none', 'train', file)
    file_path = os.path.join('dataset_apical4_none', file)
    print(file_path, train_path)
    os.rename(file_path, train_path)
for file in val_files:
    val_path = os.path.join('dataset_apical4_none', 'val', file)
    file_path = os.path.join('dataset_apical4_none', file)
    os.rename(file_path, val_path)

# Now load the annotations and create a csv with only the train and validation files
annotations = pd.read_csv('dataset_apical4_none/annotations_CO.csv')
train_annotations = annotations[annotations['patient'].isin([f[:-4] for f in train_files])]
val_annotations = annotations[annotations['patient'].isin([f[:-4] for f in val_files])]

# Rename the column patient to file_name and add the extension .mp4
train_annotations.rename(columns={'patient': 'file_name'}, inplace=True)
train_annotations = train_annotations.rename(columns={'patient': 'file_name'}).copy()
val_annotations = val_annotations.rename(columns={'patient': 'file_name'}).copy()
train_annotations['file_name'] = train_annotations['file_name'] + '.mp4'
val_annotations['file_name'] = val_annotations['file_name'] + '.mp4'
train_annotations.to_csv('dataset_apical4_none/train_annotations_CO.csv', index=False)
val_annotations.to_csv('dataset_apical4_none/val_annotations_CO.csv', index=False)
