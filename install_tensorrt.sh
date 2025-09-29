#!/bin/bash
set -e

echo "🔧 Checking for TensorRT..."
if ! python -c "import tensorrt" 2>/dev/null; then
    echo "⬇️ Installing TensorRT..."
    pip install tensorrt==10.12.0.36
else
    echo "✅ TensorRT already installed."
fi
