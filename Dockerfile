# Dockerfile
# Use an NVIDIA CUDA base image compatible with T4 GPUs and PyTorch/bitsandbytes
# CUDA 12.1.1 is a good target for recent PyTorch versions.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python, pip, and other necessary system dependencies
# Step 1: Attempt to fix persistent apt-get update issue
RUN rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/cache/apt/archives/partial \
    && apt-get update -y \
    && apt-get clean

# Step 2: Install packages with verbose output for debugging
# Adding set -x to see exactly which command is failing if Step 1 passes
RUN set -x; \
    apt-get install -y --no-install-recommends \
       python3.10 \
       python3-pip \
       python3-venv \
       git \
       curl \
       build-essential \
       libsndfile1 \
    # Clean up again to reduce image size
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create a non-root user and switch to it
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Set up a virtual environment
RUN python3.10 -m venv /home/appuser/venv
ENV PATH="/home/appuser/venv/bin:$PATH"

# Copy application files
COPY --chown=appuser:appuser requirements.txt .
COPY --chown=appuser:appuser start.sh .
COPY --chown=appuser:appuser app.py .
# If you have a .env.placeholder, copy it too for reference (though secrets are preferred on Cloud Run)
# COPY --chown=appuser:appuser .env.placeholder .

# Install PyTorch with CUDA 12.1 support first
# This ensures the correct GPU-enabled version is picked up.
# The specific versions of torch (2.7.0) and transformers (4.52.3) are from your requirements.
# If these exact versions are not available for cu121, pip will error.
RUN python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other Python dependencies from requirements.txt
# bitsandbytes will be compiled if a pre-built wheel matching the environment isn't found.
# This can take time and requires build-essential (installed above).
RUN python -m pip install --no-cache-dir -r requirements.txt

# Make start.sh executable
RUN chmod +x start.sh

# Expose the port Streamlit will run on (Cloud Run will map $PORT to this)
EXPOSE 8501

# Set the entrypoint for the container
CMD ["./start.sh"]

