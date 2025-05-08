# catdx

This repository contains a Vivit fine-tuning project on medical data.

## Need to Know
- A `.secrets` file must be placed in the repo's root directory to store your wandb API key.
- The `src` directory contains the main codebase, while the `scripts` directory contains scripts for running the project.
- The `configs` directory contains configuration files for the project.
- Set up the dataset directory and update the configuration at `src/scripts/configs/config.yaml` accordingly.
- The wandb project integration is pending (#TODO).

## Python (Single Run, Single GPU) Container
! WARNING: This container is discontiued, use the Accelerate container instead since it can run both single and multi GPU.

## Accelerate (Single Run, Multi GPU) Container
To run the script using the Accelerate container on HSSH, use the following command:
```bash
submit \
  --name vivit-contrastive \
  --gpus N \
  --mount "$(pwd)":/scratch/catdx \
  --mount /home/vitturini/shared/fix_apical4:/scratch/catdx/fix_apical4 \
  eidos-service.di.unito.it/vitturini/vivit:accelerate
```
The script will use by default every GPU allocated to the container, everything else is the same as the Python container.

## Devcontainer
Run the development container on HSSH using:
```bash
submit \
  --name vivit-dev \
  --gpus N \
  --mount /mnt/fast-scratch/vitturini/catdx:/scratch/catdx \
  --mount /home/vitturini/shared/fix_apical4:/scratch/catdx/fix_apical4 \
  eidos-service.di.unito.it/vitturini/vivit:dev
```
Check which machine your devcontainer is running on:
```bash
docker service ps my_container
```
When finished, scale the container down:
```bash
docker service scale my_container=0
```
To reconnect (scale up) with the devcontainer:
```bash
docker service scale my_container=1
```
To run the multi-gpu training script from inside the container:
```bash
accelerate launch -m src.scripts.main
```
You can add arguments to the ```accelerate launch``` command, for example:
```bash
accelerate launch --num_processes 1 --mixed_precision fp16 -m src.scripts.main
```

## Sweep Container
! WARNING: This container is not yet functional. It is a work in progress and should not be used for now.

## Building and Pushing the Container
To build the container, run the following command:
```bash
docker build -t eidos-service.di.unito.it/vitturini/vivit:base -f dockerfiles/Dockerfile_base .
```
To push the container to the repository, use:
```bash
docker push eidos-service.di.unito.it/vitturini/vivit:base
```

## Dataset format
To make the code work as it is, the dataset must be in the following format:
a dir with every video and a csv file with the labels.
```
dataset/
|-- video1.mp4
|-- video2.mp4
...
|-- videoN.mp4
|-- all_files_with_partition.csv
```

The csv file must contain the following columns:
- file_name: the name of the video file with the extension (e.g., video1.mp4)
- CO: the label for the CO
- Every other label 
- partition: the partition of the video (train, val, test)

Nevertheless, the code uses the `Huggingface Datasets` library, so it is possible to use any dataset format supported by the library with a limited amount of work. 


## WARNINGS
- A refactoring of the code is scheduled soon.
- The sweep container mounts the source code, so code should not be modified from anywhere when running the sweep.