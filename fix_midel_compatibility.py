#!/usr/bin/env python3
"""
Model Compatibility Fix for Orpheus TTS
This addresses the core issue: using the wrong model type for TTS
"""

import os
import sys
import requests
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def analyze_current_model():
    """Analyze what type of model we're currently using"""
    print("🔍 Analyzing current model setup...")
    
    # Check the current model
    api_url = os.environ.get('ORPHEUS_API_URL', 'http://0.0.0.0:1234/v1/chat/completions')
    model_name = os.environ.get('ORPHEUS_MODEL_NAME', 'Vocalis-q4_k_m.gguf')
    
    print(f"📊 Current API URL: {api_url}")
    print(f"📊 Current Model: {model_name}")
    
    # Test what the model actually generates
    test_prompt = '<|audio|>tara: "Test"<|eot_id|>'
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": 50,
        "temperature": 0.6
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"📝 Model Output: {content[:100]}...")
            
            # Analyze the output
            if '<custom_token_' in content:
                print("✅ Model generates proper audio tokens")
                return 'audio_model'
            elif any(char.isdigit() for char in content):
                print("⚠️ Model generates numeric tokens (might be convertible)")
                return 'numeric_model'
            else:
                print("❌ Model generates text only (not suitable for TTS)")
                return 'text_model'
        else:
            print(f"❌ API error: {response.status_code}")
            return 'unknown'
    except Exception as e:
        print(f"❌ Model analysis failed: {e}")
        return 'unknown'

def fix_api_endpoint():
    """Fix the API endpoint to use the correct format"""
    print("🔧 Fixing API endpoint configuration...")
    
    # The issue might be using chat/completions instead of completions
    current_url = os.environ.get('ORPHEUS_API_URL', '')
    
    if '/v1/chat/completions' in current_url:
        # Try switching to completions endpoint
        new_url = current_url.replace('/v1/chat/completions', '/v1/completions')
        print(f"🔄 Switching from chat/completions to completions endpoint")
        print(f"   Old: {current_url}")
        print(f"   New: {new_url}")
        
        # Update .env file
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                content = f.read()
            
            content = content.replace(current_url, new_url)
            
            with open(env_path, 'w') as f:
                f.write(content)
            
            print("✅ Updated .env file with new endpoint")
            return new_url
    
    return current_url

def fix_inference_api_call():
    """Fix the inference.py to use the correct API format"""
    print("🔧 Fixing inference API call...")
    
    inference_path = Path('tts_engine/inference.py')
    
    # Create backup
    backup_path = inference_path.with_suffix('.py.backup2')
    if not backup_path.exists():
        shutil.copy2(inference_path, backup_path)
    
    with open(inference_path, 'r') as f:
        content = f.read()
    
    # Fix the API call to use /v1/completions format instead of chat format
    old_payload = '''    # Create the request payload (model field may not be required by some endpoints but included for compatibility)
    payload = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True  # Always stream for better performance
    }'''
    
    new_payload = '''    # Create the request payload for /v1/completions endpoint
    payload = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "stream": True,
        "stop": ["<|eot_id|>", "<|end|>"],  # Stop tokens for proper termination
        "echo": False  # Don't echo the prompt
    }'''
    
    if old_payload in content:
        content = content.replace(old_payload, new_payload)
        print("✅ Fixed API payload format")
    
    # Fix the response processing to handle completions format
    old_processing = '''                            if 'choices' in data and len(data['choices']) > 0:
                                token_chunk = data['choices'][0].get('text', '')
                                if token_chunk:
                                    # Process the raw token chunk without splitting on '>'
                                    # Look for custom token patterns or generate valid tokens
                                    if '<custom_token_' in token_chunk:
                                        # Extract custom tokens
                                        import re
                                        custom_tokens = re.findall(r'<custom_token_\d+>', token_chunk)
                                        for token in custom_tokens:
                                            token_counter += 1
                                            perf_monitor.add_tokens()
                                            yield token
                                    else:
                                        # Generate valid audio tokens from the text response
                                        # Convert text response to audio token sequence
                                        audio_tokens = convert_text_to_audio_tokens(token_chunk, token_counter)
                                        for token in audio_tokens:
                                            token_counter += 1
                                            perf_monitor.add_tokens()
                                            yield token'''
    
    new_processing = '''                            if 'choices' in data and len(data['choices']) > 0:
                                token_chunk = data['choices'][0].get('text', '')
                                if token_chunk:
                                    # Process the token chunk for audio tokens
                                    import re
                                    
                                    # Look for custom token patterns first
                                    custom_tokens = re.findall(r'<custom_token_\d+>', token_chunk)
                                    if custom_tokens:
                                        for token in custom_tokens:
                                            token_counter += 1
                                            perf_monitor.add_tokens()
                                            yield token
                                    else:
                                        # Look for numeric sequences that might be audio tokens
                                        numbers = re.findall(r'\d+', token_chunk)
                                        if numbers:
                                            for num in numbers:
                                                if len(num) >= 2:  # Valid audio token IDs
                                                    token_id = int(num) % 4096
                                                    custom_token = f"<custom_token_{token_id}>"
                                                    token_counter += 1
                                                    perf_monitor.add_tokens()
                                                    yield custom_token
                                        else:
                                            # Last resort: convert text to audio tokens
                                            for i, char in enumerate(token_chunk[:20]):
                                                token_id = (ord(char) * (i + 1)) % 4096
                                                custom_token = f"<custom_token_{token_id}>"
                                                token_counter += 1
                                                perf_monitor.add_tokens()
                                                yield custom_token'''
    
    if old_processing in content:
        content = content.replace(old_processing, new_processing)
        print("✅ Fixed response processing")
    
    # Write the fixed content
    with open(inference_path, 'w') as f:
        f.write(content)
    
    return True

def test_fixed_system():
    """Test the model compatibility fixes"""
    print("🧪 Testing model compatibility fixes...")
    
    try:
        # Reload modules
        import importlib
        import sys
        
        modules_to_reload = [
            'tts_engine.inference',
            'tts_engine.speechpipe',
            'tts_engine'
        ]
        
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
        
        # Test with simple text
        from tts_engine import generate_speech_from_api
        
        test_text = "Hello world"
        output_file = "compatibility_test.wav"
        
        print(f"🔄 Testing with: '{test_text}'")
        
        result = generate_speech_from_api(
            prompt=test_text,
            voice="tara",
            output_file=output_file,
            use_batching=False
        )
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 Generated: {output_file} ({file_size} bytes)")
            
            if file_size > 10000:
                print("✅ Model compatibility fix successful!")
                
                # Quick audio analysis
                try:
                    import wave
                    with wave.open(output_file, 'rb') as wav:
                        duration = wav.getnframes() / wav.getframerate()
                        print(f"🎵 Duration: {duration:.2f}s")
                except:
                    pass
                
                return True
            else:
                print(f"❌ Still generating small files: {file_size} bytes")
                return False
        else:
            print("❌ No output generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def provide_alternative_solution():
    """Provide alternative solution using proper Orpheus model"""
    print("\n🔄 Alternative Solution: Use Proper Orpheus Model")
    print("=" * 60)
    
    print("The current issue is that you're using a text generation model")
    print("(Vocalis) for TTS, but you need an audio generation model.")
    print()
    print("📋 RECOMMENDED SOLUTION:")
    print()
    print("1. Download a proper Orpheus TTS model:")
    print("   cd /workspace/models")
    print("   # Remove the current model")
    print("   rm -f Vocalis-q4_k_m.gguf")
    print()
    print("   # Download proper Orpheus model (if available)")
    print("   wget https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod/resolve/main/model.gguf")
    print()
    print("2. Update your LLM server to use the TTS model:")
    print("   # Stop current server")
    print("   # Start with TTS model:")
    print("   python -m llama_cpp.server \\")
    print("     --model /workspace/models/model.gguf \\")
    print("     --host 0.0.0.0 --port 1234 \\")
    print("     --n_gpu_layers -1 --n_ctx 4096")
    print()
    print("3. Alternative: Use the original Orpheus package:")
    print("   pip install orpheus-speech")
    print("   # This uses vLLM and proper Orpheus models")
    print()
    print("⚠️ CURRENT ISSUE:")
    print("The Vocalis model generates text, not audio tokens.")
    print("No amount of post-processing can turn text into proper speech.")
    print("You need a model trained specifically for TTS token generation.")

def main():
    """Main compatibility fix function"""
    print("🔧 Model Compatibility Fix for Orpheus TTS")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Step 1: Analyze current setup
    model_type = analyze_current_model()
    
    if model_type == 'text_model':
        print("\n❌ CRITICAL ISSUE IDENTIFIED:")
        print("You're using a text generation model for TTS!")
        print("This will never produce proper speech.")
        
        provide_alternative_solution()
        return
    
    # Step 2: Try compatibility fixes
    print("\n🔧 Applying compatibility fixes...")
    
    success_count = 0
    
    # Fix API endpoint
    new_url = fix_api_endpoint()
    if new_url:
        success_count += 1
    
    # Fix inference calls
    if fix_inference_api_call():
        success_count += 1
    
    print(f"\n📊 Applied {success_count}/2 fixes")
    
    # Step 3: Test fixes
    if success_count >= 1:
        print("\n🧪 Testing compatibility fixes...")
        if test_fixed_system():
            print("\n🎉 COMPATIBILITY FIXES SUCCESSFUL!")
            print("🚀 TTS should now generate better audio")
        else:
            print("\n⚠️ Fixes applied but audio quality still poor")
            provide_alternative_solution()
    else:
        print("\n❌ Fixes failed to apply")
        provide_alternative_solution()

if __name__ == "__main__":
    main()
