# catdx
Vivit fine tuning on medical data.


## Config file
The config file is a json file that contains the following fields:
- run_name: the name of the run, is also the name of the folder where the results are saved
- dataset_folder: the path to the dataset folder
- remove_unused_columns: boolean, whether to remove unused columns
- training_mode: `contrastive` (freeze the prediction head and trains with contrastive loss), `regression` (trains only the prediction head with mse loss)
- freeze: `backbone`, `projection_head`, `classifier` or another `layer name` to freeze
- fp16: boolean, whether to use fp16
- resize_to: the size to resize the images to
- learning_rate: defaults to 5e-5
- warmup_steps: defaults to 0
- weight_decay: defaults to 0.01
- num_train_epochs: defaults to 5
- per_device_train_batch_size: defaults to 8
- per_device_eval_batch_size: defaults to 8
- tubelet_size: defaults to [2, 16, 16]
- logging_first_step: boolean, whether to log the first step
- logging_strategy: defaults to epoch, refer to huggingface documentation
- save_strategy: defaults to epoch, refer to huggingface documentation
- eval_strategy: defaults to epoch, refer to huggingface documentation
- report_to: defaults to wandb, refer to huggingface documentation


To run the script use the Dockerfile_python with the submit wrapper.
e.g.
```bash
submit --name vivit-unsup --gpus 1 --mount $(pwd):/workspace  eidos-service.di.unito.it/vitturini/vivit:python
```

for the logs use:
```bash
docker service logs -f name_of_the_container
```

Devcontainer:
```bash
submit --name vivit-dev --gpus 2 --mount /mnt/fast-scratch/vitturini/catdx:/scratch/catdx eidos-service.di.unito.it/vitturini/vivit:dev
```
