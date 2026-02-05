FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_deploy.txt .

# Install Python dependencies with version constraints
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_deploy.txt

# Clone SAM2 repository
RUN git clone https://github.com/facebookresearch/segment-anything-2.git && \
    cd segment-anything-2 && \
    pip install --no-cache-dir -e . && \
    cd ..

# Create checkpoints directory
RUN mkdir -p segment-anything-2/checkpoints

# Download SAM2 base checkpoint
RUN wget --no-check-certificate \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt \
    -O segment-anything-2/checkpoints/sam2_hiera_small.pt

# Download VREyeSAM fine-tuned weights using Python
RUN pip install --no-cache-dir huggingface-hub && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='devnagaich/VREyeSAM', \
    filename='VREyeSAM_uncertainity_best.torch', \
    local_dir='segment-anything-2/checkpoints', \
    local_dir_use_symlinks=False)"

# Verify files were downloaded
RUN ls -lh segment-anything-2/checkpoints/ && \
    test -f segment-anything-2/checkpoints/sam2_hiera_small.pt && \
    test -f segment-anything-2/checkpoints/VREyeSAM_uncertainity_best.torch && \
    echo "All checkpoints downloaded successfully!"

# Copy application files
COPY app.py .

# Expose Streamlit port
EXPOSE 7860

# Set environment variables
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
