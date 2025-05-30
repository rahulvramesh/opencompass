#!/bin/bash
set -e

# OpenCompass Docker Entrypoint Script
# This script provides flexible command execution for the OpenCompass container

# Function to print help
print_help() {
    echo "OpenCompass Docker Container"
    echo "============================"
    echo ""
    echo "Usage:"
    echo "  docker run opencompass:latest [COMMAND] [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  opencompass    - Run OpenCompass CLI (default)"
    echo "  python         - Run Python interpreter"
    echo "  bash           - Start bash shell"
    echo "  jupyter        - Start Jupyter notebook server"
    echo "  help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run evaluation"
    echo "  docker run -v \$(pwd)/data:/workspace/data opencompass:latest opencompass examples/eval_demo.py"
    echo ""
    echo "  # Interactive shell"
    echo "  docker run -it opencompass:latest bash"
    echo ""
    echo "  # Run custom Python script"
    echo "  docker run -v \$(pwd):/workspace/custom opencompass:latest python /workspace/custom/my_script.py"
    echo ""
    echo "Environment Variables:"
    echo "  HF_HOME            - Hugging Face cache directory (default: /workspace/cache)"
    echo "  TRANSFORMERS_CACHE - Transformers cache directory (default: /workspace/cache)"
    echo "  TORCH_HOME         - PyTorch cache directory (default: /workspace/cache)"
    echo ""
}

# Function to start Jupyter
start_jupyter() {
    echo "Starting Jupyter Notebook Server..."
    echo "================================="
    
    # Install jupyter if not already installed
    if ! command -v jupyter &> /dev/null; then
        echo "Installing Jupyter..."
        pip install --no-cache-dir jupyter notebook
    fi
    
    # Set default password if not provided
    JUPYTER_PASSWORD=${JUPYTER_PASSWORD:-"opencompass"}
    
    echo ""
    echo "Jupyter will be available at: http://localhost:8888"
    echo "Default password: ${JUPYTER_PASSWORD}"
    echo ""
    
    # Start Jupyter
    jupyter notebook \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --NotebookApp.password="$(python -c "from notebook.auth import passwd; print(passwd('${JUPYTER_PASSWORD}'))")" \
        --NotebookApp.token='' \
        --notebook-dir=/workspace
}

# Main entrypoint logic
case "$1" in
    "help"|"--help"|"-h")
        print_help
        ;;
    "jupyter")
        shift
        start_jupyter "$@"
        ;;
    "bash"|"sh")
        exec /bin/bash "${@:2}"
        ;;
    "python")
        exec python "${@:2}"
        ;;
    "opencompass")
        exec opencompass "${@:2}"
        ;;
    "")
        # No command provided, show help
        print_help
        ;;
    *)
        # If the first argument doesn't match any command, 
        # assume it's arguments for opencompass
        exec opencompass "$@"
        ;;
esac