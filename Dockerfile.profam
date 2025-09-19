# Use RunPod's PyTorch base image with CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Update system packages and install dependencies
RUN apt-get update && \
    apt-get install -y vim wget curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages required for ProFam
RUN pip install --no-cache-dir \
    runpod \
    torch==2.1.0 \
    lightning \
    pandas \
    numpy \
    scipy \
    transformers \
    tokenizers \
    omegaconf \
    rootutils \
    biopython \
    ml-collections \
    wandb

# Install flash-attn for efficient attention (optional but recommended)
RUN pip install --no-cache-dir flash-attn==2.3.2

# Clone the ProFam repository
RUN git clone https://github.com/alex-hh/profam

# Download the ProFam model checkpoint from HuggingFace
RUN wget -O last.ckpt "https://huggingface.co/judewells/pf/resolve/main/checkpoints/last.ckpt"

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Copy the handler script
COPY profam_handler.py .

# Set the command to run the handler
CMD ["python3", "-u", "profam_handler.py"]
