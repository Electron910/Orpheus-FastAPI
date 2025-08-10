#!/usr/bin/env python3
"""
Orpheus-FastAPI TTS Fix Script for RunPod
This script diagnoses and fixes the TTS issues causing 44-byte WAV files
"""

import os
import sys
import requests
import subprocess
import time
from pathlib import Path

def create_env_file():
    """Create the .env file with correct RunPod configuration"""
    print("📝 Creating .env configuration file...")
    
    env_content = """# Orpheus-FastAPI Configuration for RunPod
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
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def create_models_directory():
    """Create models directory and download TTS model"""
    print("📁 Creating models directory...")
    
    try:
        os.makedirs('models', exist_ok=True)
        print("✅ Models directory created")
        
        model_path = Path('models/Orpheus-3b-FT-Q8_0.gguf')
        if not model_path.exists():
            print("⬇️ Downloading Orpheus TTS model...")
            print("This may take a few minutes...")
            
            url = "https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf"
            
            # Use wget if available, otherwise use requests
            try:
                result = subprocess.run(['wget', url, '-O', str(model_path)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ TTS model downloaded successfully with wget")
                else:
                    raise Exception("wget failed")
            except:
                print("📥 Downloading with Python requests...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ TTS model downloaded successfully")
        else:
            print("✅ TTS model already exists")
        
        return True
    except Exception as e:
        print(f"❌ Error setting up models: {e}")
        return False

def verify_snac_model():
    """Verify SNAC model is available"""
    print("🔍 Verifying SNAC model...")
    
    try:
        from snac import SNAC
        model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz')
        print("✅ SNAC model is available")
        return True
    except Exception as e:
        print(f"❌ SNAC model error: {e}")
        print("🔧 Installing SNAC...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'snac==1.2.1'], check=True)
            print("✅ SNAC installed successfully")
            return True
        except Exception as install_error:
            print(f"❌ Failed to install SNAC: {install_error}")
            return False

def test_llm_api():
    """Test connectivity to the LLM API"""
    print("🌐 Testing LLM API connectivity...")
    
    try:
        response = requests.get('http://0.0.0.0:1234/v1/models', timeout=5)
        if response.status_code == 200:
            print("✅ LLM API is accessible")
            return True
        else:
            print(f"⚠️ LLM API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ LLM API not accessible: {e}")
        print("   Make sure llama-cpp-python server is running on port 1234")
        return False

def test_tts_endpoint():
    """Test the TTS endpoint"""
    print("🧪 Testing TTS endpoint...")
    
    test_payload = {
        "model": "tts-1",
        "input": "Hello, this is a test of the TTS system.",
        "voice": "tara"
    }
    
    try:
        response = requests.post(
            'http://0.0.0.0:5005/v1/audio/speech',
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            # Check if we got actual audio data
            content_length = len(response.content)
            if content_length > 1000:  # Should be much larger than 44 bytes
                print(f"✅ TTS endpoint working! Generated {content_length} bytes of audio")
                
                # Save test file
                with open('test_tts_output.wav', 'wb') as f:
                    f.write(response.content)
                print("✅ Test audio saved as test_tts_output.wav")
                return True
            else:
                print(f"❌ TTS generated only {content_length} bytes (should be >1000)")
                return False
        else:
            print(f"❌ TTS endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ TTS endpoint test failed: {e}")
        return False

def main():
    """Main setup and diagnostic function"""
    print("🔧 Orpheus-FastAPI TTS Fix Script for RunPod")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Create .env file
    if create_env_file():
        success_count += 1
    
    # Step 2: Setup models
    if create_models_directory():
        success_count += 1
    
    # Step 3: Verify SNAC
    if verify_snac_model():
        success_count += 1
    
    # Step 4: Test LLM API
    if test_llm_api():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Setup Results: {success_count}/{total_steps} steps successful")
    
    if success_count == total_steps:
        print("🎉 All setup steps completed successfully!")
        print("\n🚀 You can now start the TTS server with:")
        print("   python app.py --host 0.0.0.0 --port 5005")
        print("\n🧪 Test the TTS with:")
        print("   python fix_tts_setup.py --test")
    else:
        print("⚠️ Some setup steps failed. Please check the errors above.")
    
    # If --test flag is provided, test the endpoint
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("\n" + "=" * 50)
        print("🧪 Testing TTS Endpoint...")
        test_tts_endpoint()

if __name__ == "__main__":
    main()
