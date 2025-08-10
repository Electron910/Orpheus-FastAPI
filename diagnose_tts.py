#!/usr/bin/env python3
"""
Comprehensive TTS Diagnostic Script for RunPod
This script identifies why TTS is generating 44-byte WAV files
"""

import os
import sys
import requests
import json
import subprocess
from pathlib import Path
import wave

def check_environment_variables():
    """Check if all required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    # Load .env file if it exists
    env_file = Path('.env')
    env_vars = {}
    
    if env_file.exists():
        print("✅ .env file found")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                    os.environ[key.strip()] = value.strip()
    else:
        print("❌ .env file missing")
        return False
    
    required_vars = [
        'ORPHEUS_API_URL',
        'ORPHEUS_HOST',
        'ORPHEUS_PORT',
        'ORPHEUS_MODEL_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in env_vars:
            missing_vars.append(var)
        else:
            print(f"  ✅ {var} = {env_vars[var]}")
    
    if missing_vars:
        print(f"❌ Missing variables: {', '.join(missing_vars)}")
        return False
    
    return True

def check_models():
    """Check if required models are present"""
    print("\n🔍 Checking models...")
    
    models_dir = Path('models')
    if not models_dir.exists():
        print("❌ models/ directory missing")
        return False
    
    print("✅ models/ directory exists")
    
    # Check for TTS model
    tts_model = models_dir / 'Orpheus-3b-FT-Q8_0.gguf'
    if tts_model.exists():
        size_mb = tts_model.stat().st_size / (1024 * 1024)
        print(f"✅ TTS model found: {size_mb:.1f} MB")
    else:
        print("❌ TTS model missing: Orpheus-3b-FT-Q8_0.gguf")
        return False
    
    # Check SNAC model
    try:
        from snac import SNAC
        model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz')
        print("✅ SNAC model accessible")
    except Exception as e:
        print(f"❌ SNAC model error: {e}")
        return False
    
    return True

def check_llm_api():
    """Check LLM API connectivity and response"""
    print("\n🔍 Checking LLM API...")
    
    api_url = os.environ.get('ORPHEUS_API_URL', 'http://0.0.0.0:1234/v1/chat/completions')
    print(f"Testing API URL: {api_url}")
    
    # Test models endpoint first
    models_url = api_url.replace('/v1/chat/completions', '/v1/models')
    try:
        response = requests.get(models_url, timeout=5)
        if response.status_code == 200:
            print("✅ LLM API models endpoint accessible")
        else:
            print(f"⚠️ Models endpoint returned {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False
    
    # Test chat completions
    test_payload = {
        "model": "Vocalis-q4_k_m.gguf",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 50,
        "temperature": 0.6
    }
    
    try:
        response = requests.post(api_url, json=test_payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                print(f"✅ LLM API working, response: {content[:50]}...")
                return True
            else:
                print("❌ LLM API returned empty response")
                return False
        else:
            print(f"❌ LLM API returned {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ LLM API test failed: {e}")
        return False

def test_tts_generation():
    """Test actual TTS generation step by step"""
    print("\n🔍 Testing TTS generation process...")
    
    # Import TTS modules
    try:
        from tts_engine import generate_speech_from_api
        print("✅ TTS engine imported successfully")
    except Exception as e:
        print(f"❌ TTS engine import failed: {e}")
        return False
    
    # Test with simple text
    test_text = "Hello, this is a test."
    output_file = "diagnostic_test.wav"
    
    print(f"Generating speech for: '{test_text}'")
    
    try:
        # Generate speech
        result = generate_speech_from_api(
            prompt=test_text,
            voice="tara",
            output_file=output_file,
            use_batching=False
        )
        
        # Check output file
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 Output file size: {file_size} bytes")
            
            if file_size > 1000:
                print("✅ TTS generation successful!")
                
                # Analyze WAV file
                try:
                    with wave.open(output_file, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        duration = frames / sample_rate
                        print(f"🎵 Audio duration: {duration:.2f} seconds")
                        print(f"🎵 Sample rate: {sample_rate} Hz")
                        print(f"🎵 Channels: {wav_file.getnchannels()}")
                except Exception as e:
                    print(f"⚠️ WAV file analysis failed: {e}")
                
                return True
            else:
                print("❌ TTS generated empty/small file (likely the 44-byte issue)")
                
                # Try to read the small file to see what's in it
                try:
                    with open(output_file, 'rb') as f:
                        content = f.read()
                        print(f"File content (first 44 bytes): {content[:44]}")
                except:
                    pass
                
                return False
        else:
            print("❌ No output file generated")
            return False
            
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_diagnostic():
    """Run complete diagnostic suite"""
    print("🔧 Orpheus-FastAPI TTS Diagnostic Tool")
    print("=" * 50)
    
    results = {
        'environment': check_environment_variables(),
        'models': check_models(),
        'llm_api': check_llm_api(),
        'tts_generation': test_tts_generation()
    }
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC RESULTS:")
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test.upper()}: {status}")
    
    failed_tests = [test for test, passed in results.items() if not passed]
    
    if not failed_tests:
        print("\n🎉 All tests passed! TTS should be working.")
    else:
        print(f"\n⚠️ Failed tests: {', '.join(failed_tests)}")
        print("\n🔧 RECOMMENDED FIXES:")
        
        if not results['environment']:
            print("  1. Run: python fix_tts_setup.py")
        
        if not results['models']:
            print("  2. Download models: bash setup_runpod_tts.sh")
        
        if not results['llm_api']:
            print("  3. Check LLM server is running on port 1234")
            print("     Command: python -m llama_cpp.server --model /workspace/models/Vocalis-q4_k_m.gguf --host 0.0.0.0 --port 1234 --n_gpu_layers -1")
        
        if not results['tts_generation']:
            print("  4. Check TTS engine logs for detailed errors")

if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    run_full_diagnostic()
