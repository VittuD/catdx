# catdx

This repository contains a Vivit fine-tuning project on medical data.

## Need to Know
- A `.secrets` file must be placed in the repo's root directory to store your wandb API key.
- The `src` directory contains the main codebase, while the `scripts` directory contains scripts for running the project.
- The `configs` directory contains configuration files for the project.
- Set up the dataset directory and update the configuration at `src/scripts/configs/config.yaml` accordingly.
- The wandb project integration is pending (#TODO).

## Python (Single Run) Container
To run the script using the Python container on HSSH, use the following command:
```bash
submit --name my_container --gpus N --mount $(pwd):/workspace eidos-service.di.unito.it/vitturini/vivit:python
```
View logs with:
```bash
docker service logs -f name_of_the_container
```
To remove the container when finished:
```bash
docker service rm my_container
```

## Devcontainer
Run the development container on HSSH using:
```bash
submit --name my_container-dev --gpus N --mount /mnt/fast-scratch/vitturini/catdx:/scratch/catdx eidos-service.di.unito.it/vitturini/vivit:dev
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

## Sweep Container
! WARNING: This container is not yet functional. It is a work in progress and should not be used for now.

## WARNINGS
- A refactoring of the code is scheduled soon.
- The sweep container mounts the source code, so code should not be modified from anywhere when running the sweep.
