import os
import cv2
import pydicom
import shutil

def convert_video(file_path, output_mp4):
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    tmp_folder = os.path.join("/tmp", base_filename)
    os.makedirs(tmp_folder, exist_ok=True)
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".dcm", ".dicomdir", ""]:
        # Process DICOM file
        dcm = pydicom.dcmread(file_path)
        if 'PixelData' not in dcm:
            print(f"No pixel data in {file_path}")
            shutil.rmtree(tmp_folder)
            return
        frames = dcm.pixel_array
        # Use FrameTime (in ms) if available, otherwise default to 30 FPS
        fps = 1000 / float(dcm.FrameTime) if hasattr(dcm, 'FrameTime') else 30
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(tmp_folder, f'frame_{i}.png'), frame)
            
    elif ext == ".avi":
        # Process AVI file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error opening {file_path}")
            shutil.rmtree(tmp_folder)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(tmp_folder, f'frame_{count}.png'), frame)
            count += 1
        cap.release()
        
    else:
        print(f"Unsupported file format: {file_path}")
        shutil.rmtree(tmp_folder)
        return

    # Assemble extracted frames into an MP4 video
    frames_list = sorted([f for f in os.listdir(tmp_folder) if f.endswith('.png')],
                          key=lambda x: int(x.split('_')[1].split('.')[0]))
    if not frames_list:
        print(f"No frames extracted from {file_path}")
        shutil.rmtree(tmp_folder)
        return

    first_frame = cv2.imread(os.path.join(tmp_folder, frames_list[0]))
    height, width = first_frame.shape[:2]
    out = cv2.VideoWriter(output_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for fname in frames_list:
        frame = cv2.imread(os.path.join(tmp_folder, fname))
        out.write(frame)
    out.release()
    shutil.rmtree(tmp_folder)
    print(f"Converted {file_path} to {output_mp4}")

def convert_to_bw(input_mp4, output_mp4):
    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        print(f"Error opening video file {input_mp4}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a VideoWriter for grayscale output.
    out = cv2.VideoWriter(output_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
    
    cap.release()
    out.release()
    print(f"Converted {input_mp4} to black and white video {output_mp4}")

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(input_dir):
        file_path = os.path.join(input_dir, item)
        if os.path.isfile(file_path):
            # If no extension, assume DICOM
            ext = os.path.splitext(item)[1].lower() or ".dcm"
            if ext in [".dcm", ".dicomdir", ".avi"]:
                output_mp4 = os.path.join(output_dir, os.path.splitext(item)[0] + ".mp4")
                convert_video(file_path, output_mp4)
            else:
                print(f"Skipping unsupported file: {item}")

if __name__ == '__main__':
    input_directory = "/scratch/catdx/preprocessed/apical4"
    output_directory = "/scratch/catdx/preprocessed/mp4/apical4"
    main(input_directory, output_directory)
