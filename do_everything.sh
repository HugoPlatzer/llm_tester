#!/bin/bash
set -e

# Check for correct number of arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config.json>"
    exit 1
fi

CONFIG_FILE="$1"

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

# Build llama.cpp
echo "Building llama.cpp..."
if [ -d "llama.cpp" ]; then
    echo "Removing existing llama.cpp directory..."
    rm -rf llama.cpp
fi

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
cd ..

# Generate output filename
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .json)
RESULTS_JSON="${CONFIG_BASENAME}_results.json"

# Run main script
echo "Starting model processing..."
python3 "$SCRIPT_DIR/main.py" \
    --llama-run-path "llama.cpp/build/bin/llama-run" \
    --config "$CONFIG_FILE" \
    --output "$RESULTS_JSON"

echo "Processing complete! Results saved to: $RESULTS_JSON"
