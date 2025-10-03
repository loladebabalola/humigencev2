#!/bin/bash
# Setup script for llama-server (llama.cpp server)

echo "🚀 Setting up llama-server for Multi-Tenant Inference"
echo "=================================================="

# Check if llama.cpp is already installed
if command -v llama-server &> /dev/null; then
    echo "✅ llama-server is already installed"
    llama-server --version
    exit 0
fi

# Check if we're in the right directory
if [ ! -d "llama.cpp" ]; then
    echo "📥 Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

echo "🔧 Building llama.cpp with CUDA support..."

# Build with CUDA support
make clean
make -j$(nproc) LLAMA_CUDA=1

# Install the server binary
echo "📦 Installing llama-server..."
sudo cp server /usr/local/bin/llama-server
sudo chmod +x /usr/local/bin/llama-server

# Verify installation
if command -v llama-server &> /dev/null; then
    echo "✅ llama-server installed successfully!"
    llama-server --version
else
    echo "❌ Installation failed"
    exit 1
fi

echo ""
echo "🎉 Setup complete! You can now use Multi-Tenant Inference."
echo ""
echo "Next steps:"
echo "1. Place your GGUF model in /models/gpt-oss-20b/"
echo "2. Run the Multi-Tenant Inference wizard again"
echo "3. Start the services"
