FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_deploy.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Clone SAM2 repository
RUN git clone https://github.com/facebookresearch/segment-anything-2.git && \
    cd segment-anything-2 && \
    pip install --no-cache-dir -e .

# Download SAM2 checkpoint
RUN mkdir -p segment-anything-2/checkpoints && \
    cd segment-anything-2/checkpoints && \
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

# Download VREyeSAM fine-tuned weights from Hugging Face
RUN pip install --no-cache-dir huggingface-hub && \
    huggingface-cli download devnagaich/VREyeSAM VREyeSAM_uncertainity_best.torch \
    --local-dir segment-anything-2/checkpoints/

# Copy application files
COPY app.py .

# Expose Streamlit port
EXPOSE 7860

# Set environment variables
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]