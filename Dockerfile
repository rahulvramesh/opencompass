# Multi-stage build for OpenCompass
# Build arguments
ARG PYTHON_VERSION=3.10
ARG BUILD_DATE
ARG VERSION
ARG REVISION

# Stage 1: Base image with system dependencies
FROM python:${PYTHON_VERSION}-slim AS base

# Labels
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.revision="${REVISION}"
LABEL org.opencontainers.image.title="OpenCompass"
LABEL org.opencontainers.image.description="A unified evaluation platform for large language models"
LABEL org.opencontainers.image.source="https://github.com/open-compass/opencompass"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libgl1 \
    libglew-dev \
    libosmesa6-dev \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/* 

# Stage 2: Builder stage for dependencies
FROM base AS builder

WORKDIR /tmp

# Copy requirements files
COPY requirements/ /tmp/requirements/

# Install PyTorch (CPU version for CI/CD)
# For CUDA support, change the index URL to the appropriate CUDA version
RUN pip install --no-cache-dir torch>=1.13.1 --index-url https://download.pytorch.org/whl/cpu

# Install base runtime requirements
RUN pip install --no-cache-dir -r requirements/runtime.txt

# Install API requirements (optional)
RUN pip install --no-cache-dir -r requirements/api.txt || true

RUN pip install --no-cache-dir -r requirements/extra.txt || true

RUN pip install --no-cache-dir latex2sympy2-extended

# Stage 3: Final runtime image
FROM base AS runtime

# Set working directory
WORKDIR /workspace

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the entire project
COPY . /workspace/

# Install OpenCompass in development mode
RUN pip install math_verify latex2sympy2_extended
RUN pip install -e .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/outputs /workspace/cache

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/workspace/cache
ENV TRANSFORMERS_CACHE=/workspace/cache
ENV TORCH_HOME=/workspace/cache

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose port for potential API services
EXPOSE 8000

# Expose Jupyter port
EXPOSE 8888

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["help"]

# Alternative Dockerfile for development with all optional dependencies
# Uncomment the following to build a development image with all features

# FROM runtime as dev
# 
# # Install development and testing requirements
# RUN pip install --no-cache-dir -r requirements/dev.txt || true
# RUN pip install --no-cache-dir -r requirements/test.txt || true
# 
# # Install optional inference backends (choose one)
# # RUN pip install --no-cache-dir -r requirements/lmdeploy.txt
# # RUN pip install --no-cache-dir -r requirements/vllm.txt
# 
# # Install extra evaluation tools

# 
# CMD ["bash"]
