#!/bin/bash

# Orpheus-FastAPI RunPod Setup Script
# This script fixes the TTS issues by setting up proper environment and models

echo "🔧 Setting up Orpheus-FastAPI for RunPod..."

# Create the .env file (bypassing gitignore)
echo "📝 Creating .env configuration file..."
cat > .env << 'EOF'
# Orpheus-FastAPI Configuration for RunPod
# Server connection settings - Use 0.0.0.0 for RunPod deployment
ORPHEUS_API_URL=http://0.0.0.0:1234/v1/chat/completions
ORPHEUS_API_TIMEOUT=120

# Generation parameters
ORPHEUS_MAX_TOKENS=8192
ORPHEUS_TEMPERATURE=0.6
ORPHEUS_TOP_P=0.9
ORPHEUS_SAMPLE_RATE=24000
ORPHEUS_MODEL_NAME=Orpheus-3b-FT-Q8_0.gguf

# Web UI settings
ORPHEUS_PORT=5005
ORPHEUS_HOST=0.0.0.0
EOF

# Create models directory (bypassing gitignore)
echo "📁 Creating models directory..."
mkdir -p models

# Download the Orpheus TTS model if not present
echo "📥 Checking for Orpheus TTS model..."
if [ ! -f "models/Orpheus-3b-FT-Q8_0.gguf" ]; then
    echo "⬇️ Downloading Orpheus TTS model..."
    cd models
    wget https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf
    cd ..
    echo "✅ TTS model downloaded successfully"
else
    echo "✅ TTS model already exists"
fi

# Verify SNAC model is available (should be downloaded by pip install)
echo "🔍 Verifying SNAC model..."
python3 -c "
try:
    from snac import SNAC
    model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz')
    print('✅ SNAC model is available')
except Exception as e:
    print(f'❌ SNAC model error: {e}')
    print('Installing SNAC model...')
    import subprocess
    subprocess.run(['pip', 'install', 'snac==1.2.1'])
"

# Test the API URL connectivity
echo "🌐 Testing LLM API connectivity..."
curl -s --max-time 5 http://0.0.0.0:1234/v1/models > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ LLM API is accessible"
else
    echo "⚠️ LLM API not accessible - make sure llama-cpp-python server is running on port 1234"
fi

# Set proper permissions
chmod 644 .env
chmod -R 755 models/

echo "🎉 Setup complete!"
echo ""
echo "📋 Summary of changes:"
echo "  ✅ Created .env file with correct RunPod configuration"
echo "  ✅ Created models directory"
echo "  ✅ Downloaded Orpheus TTS model (if missing)"
echo "  ✅ Verified SNAC model availability"
echo ""
echo "🚀 You can now start the TTS server with:"
echo "  python app.py --host 0.0.0.0 --port 5005"
echo ""
echo "🧪 Test the TTS endpoint with:"
echo "  curl -X POST http://0.0.0.0:5005/v1/audio/speech \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"model\": \"tts-1\", \"input\": \"Hello, this is a test.\", \"voice\": \"tara\"}' \\"
echo "    --output test_output.wav"
