# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git and clean up in the same layer
RUN apt-get update && apt-get install -y git ninja-build && rm -rf /var/lib/apt/lists/*

# Update pip and uninstall apex in the same layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y apex

# Install flash-attention separately first to ensure proper CUDA compilation
RUN pip install packaging && \
    pip install flash-attn==2.6.3 --no-build-isolation

# Install xfuser without flash-attn (we installed it separately)
RUN pip install "xfuser[diffusers]" flask

# Copy only the necessary example script
COPY ./examples/run_flux_docker.py /app/run_flux_docker.py

# Create output directory
RUN mkdir -p /outputs

# Set working directory
WORKDIR /app

# Set default command to run the flux example
ENTRYPOINT ["python", "run_flux_docker.py"]
