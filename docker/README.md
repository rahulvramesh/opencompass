# OpenCompass Docker Setup

This directory contains Docker configuration files for running OpenCompass in containerized environments.

## Quick Start

### Building the Docker Image

```bash
# Build the base image
docker build -t opencompass:latest .

# Build with GPU support (requires nvidia-docker)
docker build -t opencompass:gpu --build-arg CUDA_VERSION=11.8 .
```

### Running with Docker

```bash
# Show help
docker run --rm opencompass:latest

# Run evaluation
docker run --rm -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    opencompass:latest opencompass examples/eval_demo.py

# Interactive shell
docker run -it --rm opencompass:latest bash

# With GPU support
docker run --gpus all -it --rm opencompass:latest bash
```

### Using Docker Compose

```bash
# Start the service
docker-compose up -d opencompass

# Run commands
docker-compose exec opencompass opencompass examples/eval_demo.py

# Start development environment
docker-compose up -d opencompass-dev

# Start Jupyter notebook
docker-compose up -d jupyter
```

## Configuration

### Environment Variables

- `HF_HOME`: Hugging Face cache directory (default: `/workspace/cache`)
- `TRANSFORMERS_CACHE`: Transformers cache directory (default: `/workspace/cache`)
- `TORCH_HOME`: PyTorch cache directory (default: `/workspace/cache`)
- `TOKENIZERS_PARALLELISM`: Set to `false` to disable warning
- API Keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

### Volumes

- `/workspace/data`: Input data directory
- `/workspace/outputs`: Output results directory
- `/workspace/cache`: Model cache directory
- `/workspace/configs`: Configuration files (read-only by default)

### Dockerfile Targets

- `base`: Base image with system dependencies
- `builder`: Build stage for installing Python packages
- `runtime`: Final runtime image (default)
- `dev`: Development image with all optional dependencies

## Examples

### Basic Evaluation

```bash
docker run --rm \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/cache:/workspace/cache \
    opencompass:latest \
    opencompass configs/eval_demo.py
```

### With API Keys

```bash
docker run --rm \
    -e OPENAI_API_KEY="your-key-here" \
    -v $(pwd)/outputs:/workspace/outputs \
    opencompass:latest \
    opencompass configs/eval_gpt3.5.py
```

### Development Mode

```bash
# Mount source code for live editing
docker run -it --rm \
    -v $(pwd):/workspace \
    -v $(pwd)/cache:/workspace/cache \
    opencompass:latest bash
```

### Jupyter Notebook

```bash
docker run -d \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    opencompass:latest jupyter

# Access at http://localhost:8888
# Default password: opencompass
```

## GPU Support

For GPU support, ensure you have:
1. NVIDIA Docker runtime installed
2. Compatible CUDA drivers

```bash
# Run with all GPUs
docker run --gpus all opencompass:latest

# Run with specific GPUs
docker run --gpus '"device=0,1"' opencompass:latest

# Using docker-compose
docker-compose --profile gpu up
```

## Troubleshooting

### Out of Memory

Increase Docker memory limits in Docker Desktop settings or use:
```bash
docker run --memory="8g" --memory-swap="16g" opencompass:latest
```

### Permission Issues

If you encounter permission issues with mounted volumes:
```bash
# Run with user ID mapping
docker run --user $(id -u):$(id -g) opencompass:latest
```

### Slow Model Downloads

Mount a persistent cache directory:
```bash
mkdir -p ./cache
docker run -v $(pwd)/cache:/workspace/cache opencompass:latest
```