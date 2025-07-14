FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# Set working directory
WORKDIR /app

# Copy objective function file
COPY objective_fn.py .

# Install required dependencies
RUN pip install --no-cache-dir \
    torch torchvision tqdm \
    kubeflow-katib kubernetes

# Optional: Set this so it's usable in Python modules
ENV PYTHONPATH="/app"

# No entrypoint needed, will be used by Elyra as a utility container
