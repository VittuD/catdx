# Base NVIDIA CUDA image with PyTorch and Python pre-installed
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set up a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

# Create a non-root user
RUN apt-get update && apt-get install -y \
    sudo \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && chmod 0440 /etc/sudoers

# Switch to the non-root user
USER $USERNAME

# Add local bin to PATH for user-installed Python packages
ENV PATH="$PATH:/home/$USERNAME/.local/bin"

# Create a workspace directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install accelerate
RUN pip install --no-cache-dir --user accelerate

# Copy accelerate configuration directly to the user's configuration directory
RUN mkdir -p /home/$USERNAME/.cache/huggingface/accelerate
COPY accelerate_single_config.yaml /home/$USERNAME/.cache/huggingface/accelerate/default_config.yaml

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Default command
CMD ["accelerate", "launch", "main.py"]
