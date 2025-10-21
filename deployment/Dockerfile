# LeapHand Planner V4 Deployment Dockerfile

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    onnxruntime \
    onnxruntime-gpu \
    torch \
    torchvision

# Create app directory
WORKDIR /app

# Copy model and inference script
COPY leap_hand_v4.onnx ./model.onnx
COPY leap_hand_v4_inference.py ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port (if needed for API)
EXPOSE 8000

# Default command
CMD ["python", "leap_hand_v4_inference.py", "--model_path", "model.onnx"]
