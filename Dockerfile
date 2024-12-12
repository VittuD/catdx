# Base NVIDIA CUDA image with PyTorch and Python pre-installed
FROM eidos-service.di.unito.it/eidos-base-pytorch:2.5.1

# Copy only the requirements.txt file to the container
COPY requirements.txt /workspace/requirements.txt

# Install dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy accelerate configuration file
COPY accelerate_multi_config.yaml /home/eidos/.cache/huggingface/accelerate/default_config.yaml

# Set the working directory
WORKDIR /workspace

# Set vscode as the entrypoint
ENTRYPOINT ["accelerate", "launch", "main.py"]
