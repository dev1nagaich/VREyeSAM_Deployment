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

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_deploy.txt

# Clone SAM2 repository at a specific commit that has hieradet
# Using an older version that's compatible with the VREyeSAM weights
RUN git clone https://github.com/facebookresearch/segment-anything-2.git && \
    cd segment-anything-2 && \
    git checkout 7e1596c0b6462eb1d1ba7e1492430fed95023598 && \
    pip install --no-cache-dir -e . && \
    cd ..

# Create checkpoints directory
RUN mkdir -p segment-anything-2/checkpoints

# Download SAM2 base checkpoint
RUN wget --no-check-certificate \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt \
    -O segment-anything-2/checkpoints/sam2_hiera_small.pt

# Download VREyeSAM weights using direct wget from HuggingFace
RUN wget --no-check-certificate \
    "https://huggingface.co/devnagaich/VREyeSAM/resolve/main/VREyeSAM_uncertainity_best.torch" \
    -O segment-anything-2/checkpoints/VREyeSAM_uncertainity_best.torch

# Verify files were downloaded
RUN ls -lh segment-anything-2/checkpoints/ && \
    test -f segment-anything-2/checkpoints/sam2_hiera_small.pt && \
    test -f segment-anything-2/checkpoints/VREyeSAM_uncertainity_best.torch

# Create Streamlit config directory
RUN mkdir -p /root/.streamlit

# Copy Streamlit config to increase upload size
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Copy application files
COPY app.py .

# Expose Streamlit port
EXPOSE 7860

# Set environment variables
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.maxUploadSize=500"]
