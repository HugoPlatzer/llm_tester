#!/bin/bash
set -e

# Check for correct number of arguments
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <config.json> [--cuda]"
    exit 1
fi

CONFIG_FILE="$1"
USE_CUDA=0

# Check for --cuda flag
if [ $# -eq 2 ]; then
    if [ "$2" = "--cuda" ]; then
        USE_CUDA=1
    else
        echo "Error: Invalid argument $2. Usage: $0 <config.json> [--cuda]"
        exit 1
    fi
fi

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found."
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Install aria2
echo "Installing aria2..."
apt install -y aria2

# Build llama.cpp
echo "Building llama.cpp..."
if [ -d "llama.cpp" ]; then
    echo "Removing existing llama.cpp directory..."
    rm -rf llama.cpp
fi

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Add CUDA option if flag is set
if [ $USE_CUDA -eq 1 ]; then
    cmake -B build -DGGML_CUDA=ON
else
    cmake -B build
fi

cmake --build build --config Release
cd ..

# Generate output filename
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .json)
RESULTS_JSON="${CONFIG_BASENAME}_results.json"

# Run main script with optional CUDA flag
PYTHON_ARGS=(
    --llama-run-path "llama.cpp/build/bin/llama-run"
    --config "$CONFIG_FILE"
    --output "$RESULTS_JSON"
)

if [ $USE_CUDA -eq 1 ]; then
    PYTHON_ARGS+=(--cuda)
fi

echo "Starting model processing..."
python3 "$SCRIPT_DIR/main2.py" "${PYTHON_ARGS[@]}"

echo "Processing complete! Results saved to: $RESULTS_JSON"
