#!/usr/bin/env python3
"""
Comprehensive Setup Script for Original Orpheus TTS
This script sets up the environment and tests the original Orpheus TTS model
"""

import os
import sys
import subprocess
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"🔧 {description}")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ Failed with return code {result.returncode}")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return False

def check_huggingface_auth():
    """Check if user is authenticated with Hugging Face"""
    print("🔐 Checking Hugging Face authentication...")
    
    try:
        import huggingface_hub
        token = huggingface_hub.get_token()
        if token:
            print("   ✅ Hugging Face token found")
            return True
        else:
            print("   ❌ No Hugging Face token found")
            return False
    except Exception as e:
        print(f"   ❌ Error checking auth: {str(e)}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "pip install --upgrade pip",
        "pip install orpheus-speech",
        "pip install flask fastapi uvicorn",
        "pip install python-dotenv",
        "pip install wave struct"
    ]
    
    for dep in dependencies:
        if not run_command(dep, f"Installing: {dep.split()[-1]}"):
            print(f"   ⚠️ Warning: Failed to install {dep}")
    
    return True

def create_env_file():
    """Create .env file with proper configuration"""
    print("📝 Creating .env file...")
    
    env_content = """# Orpheus TTS Configuration
ORPHEUS_MODEL_NAME=canopylabs/orpheus-tts-0.1-finetune-prod
TTS_HOST=0.0.0.0
TTS_PORT=5005
DEBUG=false

# Original API configuration (for compatibility)
ORPHEUS_API_URL=http://0.0.0.0:1234/v1/chat/completions
ORPHEUS_MAX_TOKENS=2000
ORPHEUS_TEMPERATURE=0.4
ORPHEUS_TOP_P=0.9
ORPHEUS_REPETITION_PENALTY=1.1
"""
    
    env_path = Path(".env")
    try:
        with open(env_path, "w") as f:
            f.write(env_content)
        print("   ✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create .env file: {str(e)}")
        return False

def test_original_orpheus():
    """Test the original Orpheus TTS model"""
    print("🎤 Testing Original Orpheus TTS...")
    
    try:
        # Set multiprocessing method
        multiprocessing.set_start_method('spawn', force=True)
        
        from orpheus_tts import OrpheusModel
        import wave
        import time
        
        print("   📥 Loading Orpheus model...")
        model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        print("   ✅ Model loaded successfully!")
        
        prompt = "Hello, this is a test of the original Orpheus TTS system."
        
        print(f"   🗣️ Generating speech for: '{prompt}'")
        start_time = time.monotonic()
        
        # Generate speech tokens
        syn_tokens = model.generate_speech(
            prompt=prompt,
            voice="tara",
            repetition_penalty=1.1,
            max_tokens=500,  # Smaller for testing
            temperature=0.4,
            top_p=0.9
        )
        
        # Save to WAV file
        output_file = "orpheus_test_output.wav"
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            total_frames = 0
            chunk_count = 0
            
            for audio_chunk in syn_tokens:
                if audio_chunk is not None and len(audio_chunk) > 0:
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    chunk_count += 1
                    wf.writeframes(audio_chunk)
            
            duration = total_frames / wf.getframerate()
            end_time = time.monotonic()
            
            print(f"   🎵 Generated {duration:.2f} seconds of audio in {end_time - start_time:.2f} seconds")
            print(f"   📊 Total chunks: {chunk_count}, Total frames: {total_frames}")
            
            # Check file size
            file_size = os.path.getsize(output_file)
            print(f"   📁 Output file: {output_file} ({file_size} bytes)")
            
            if file_size > 1000:  # More than just WAV header
                print("   ✅ SUCCESS: Original Orpheus TTS is working!")
                return True
            else:
                print("   ❌ WARNING: Generated file is too small")
                return False
                
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup function"""
    print("🚀 Orpheus TTS Original Model Setup")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Check HuggingFace auth
    if check_huggingface_auth():
        success_count += 1
    else:
        print("   ⚠️ Please run 'huggingface-cli login' first!")
    
    # Step 2: Install dependencies
    if install_dependencies():
        success_count += 1
    
    # Step 3: Create .env file
    if create_env_file():
        success_count += 1
    
    # Step 4: Test original Orpheus
    if test_original_orpheus():
        success_count += 1
    
    # Step 5: Final summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)
    
    if success_count >= 4:
        print("🎉 SUCCESS: Original Orpheus TTS is ready!")
        print("\n📋 Next Steps:")
        print("1. Run the hybrid TTS server:")
        print("   python hybrid_tts_app.py")
        print("\n2. Or run the streaming server:")
        print("   python streaming_tts_server.py")
        print("\n3. Test the API:")
        print("   curl 'http://localhost:5005/tts?prompt=Hello%20world'")
        print("\n4. View the web interface:")
        print("   http://localhost:5005/")
        success_count += 1
    else:
        print("❌ SETUP INCOMPLETE")
        print(f"   Completed: {success_count}/{total_steps} steps")
        print("\n🔍 Issues to resolve:")
        if success_count < 1:
            print("   - Hugging Face authentication required")
        if success_count < 2:
            print("   - Dependency installation failed")
        if success_count < 3:
            print("   - Environment configuration failed")
        if success_count < 4:
            print("   - Orpheus TTS model test failed")
    
    print("\n" + "=" * 50)
    return success_count >= 4

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
