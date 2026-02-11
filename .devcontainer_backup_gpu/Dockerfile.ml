# Start from a clean, official NVIDIA image with CUDA 12.4
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install the Python libraries for ML
# We install torch first with the correct CUDA index
RUN pip install \
    torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the ML stack
RUN pip install \
    sentence-transformers \
    transformers \
    flask

# Copy our server script into the container
WORKDIR /app
COPY ml-worker.py .

# Expose the port the server will run on
EXPOSE 5001

# Start the server when the container launches
ENTRYPOINT ["python3", "ml-worker.py"]
