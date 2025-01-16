# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git and clean up in the same layer
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Update pip and uninstall apex in the same layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y apex

# Install xfuser with all extras
RUN pip install "xfuser[diffusers,flash-attn]" flask

# Copy only the necessary example script
COPY ./examples/run_flux_docker.py /app/run_flux_docker.py

# Create output directory
RUN mkdir -p /outputs

# Set working directory
WORKDIR /app

# Set default command to run the flux example
ENTRYPOINT ["python", "run_flux_docker.py"]
