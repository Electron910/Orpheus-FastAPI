#!/usr/bin/env python3
"""
Token Debug Script for Orpheus TTS
This script analyzes the token generation and conversion process to identify why audio conversion fails
"""

import os
import sys
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_token_generation():
    """Test token generation from LLM and analyze the output"""
    print("🔍 Testing token generation from LLM...")
    
    api_url = os.environ.get('ORPHEUS_API_URL', 'http://0.0.0.0:1234/v1/chat/completions')
    
    # Test with the exact format used by the TTS system
    test_prompt = '<|audio|>tara: "Hello, this is a test."<|eot_id|>'
    
    payload = {
        "model": "Vocalis-q4_k_m.gguf",
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": 100,
        "temperature": 0.6,
        "top_p": 0.9,
        "stream": False
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"✅ LLM Response: {content[:100]}...")
            
            # Analyze the content for token patterns
            print(f"📊 Response length: {len(content)} characters")
            
            # Check for audio token patterns
            if '<|audio|>' in content:
                print("✅ Found <|audio|> tokens in response")
            else:
                print("❌ No <|audio|> tokens found")
            
            # Look for numeric patterns that might be audio tokens
            import re
            numbers = re.findall(r'\d+', content)
            if numbers:
                print(f"🔢 Found {len(numbers)} numeric values: {numbers[:10]}...")
            else:
                print("❌ No numeric values found in response")
            
            return content
        else:
            print(f"❌ LLM API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return None

def test_token_parsing():
    """Test the token parsing and conversion process"""
    print("\n🔍 Testing token parsing process...")
    
    try:
        # Import the TTS engine modules
        sys.path.append('.')
        from tts_engine.inference import generate_tokens_from_api, format_prompt
        from tts_engine.speechpipe import turn_token_into_id
        
        print("✅ TTS modules imported successfully")
        
        # Test prompt formatting
        test_text = "Hello, this is a test."
        formatted_prompt = format_prompt(test_text, "tara")
        print(f"📝 Formatted prompt: {formatted_prompt}")
        
        # Test token generation (just a few tokens)
        print("🔄 Generating tokens from API...")
        token_gen = generate_tokens_from_api(
            prompt=test_text,
            voice="tara",
            max_tokens=50  # Limit for testing
        )
        
        # Collect first few tokens
        tokens = []
        token_count = 0
        for token in token_gen:
            tokens.append(token)
            token_count += 1
            if token_count >= 20:  # Just get first 20 tokens
                break
        
        print(f"📊 Generated {len(tokens)} tokens: {tokens[:10]}...")
        
        # Test token-to-ID conversion
        print("🔄 Testing token-to-ID conversion...")
        valid_ids = []
        invalid_tokens = []
        
        for i, token in enumerate(tokens[:10]):
            token_id = turn_token_into_id(str(token), i)
            if token_id is not None:
                valid_ids.append(token_id)
                print(f"  Token '{token}' -> ID {token_id}")
            else:
                invalid_tokens.append(token)
                print(f"  Token '{token}' -> INVALID")
        
        print(f"📊 Valid IDs: {len(valid_ids)}/{len(tokens[:10])}")
        print(f"📊 Valid ID range: {min(valid_ids) if valid_ids else 'N/A'} - {max(valid_ids) if valid_ids else 'N/A'}")
        
        if invalid_tokens:
            print(f"❌ Invalid tokens: {invalid_tokens}")
        
        return len(valid_ids) > 0
        
    except Exception as e:
        print(f"❌ Token parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_snac_conversion():
    """Test SNAC model conversion with sample data"""
    print("\n🔍 Testing SNAC conversion...")
    
    try:
        from snac import SNAC
        import torch
        import numpy as np
        
        # Load SNAC model
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"✅ SNAC model loaded on {device}")
        
        # Test with sample valid codes (from SNAC documentation)
        print("🧪 Testing with sample valid codes...")
        
        # Create sample codes in the expected format
        sample_codes = [
            torch.randint(0, 4096, (1, 10), device=device),  # Code 0
            torch.randint(0, 4096, (1, 20), device=device),  # Code 1  
            torch.randint(0, 4096, (1, 40), device=device)   # Code 2
        ]
        
        with torch.inference_mode():
            audio_output = model.decode(sample_codes)
            
        if audio_output is not None and len(audio_output) > 0:
            audio_np = audio_output.cpu().numpy()
            print(f"✅ SNAC conversion successful: {audio_np.shape}")
            print(f"📊 Audio range: {audio_np.min():.3f} to {audio_np.max():.3f}")
            return True
        else:
            print("❌ SNAC conversion returned empty audio")
            return False
            
    except Exception as e:
        print(f"❌ SNAC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_existing_wav_file():
    """Analyze the 44-byte WAV files to understand what's in them"""
    print("\n🔍 Analyzing existing WAV files...")
    
    outputs_dir = Path('outputs')
    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return
    
    wav_files = list(outputs_dir.glob('*.wav'))
    if not wav_files:
        print("❌ No WAV files found in outputs/")
        return
    
    # Analyze the most recent WAV file
    latest_wav = max(wav_files, key=lambda f: f.stat().st_mtime)
    file_size = latest_wav.stat().st_size
    
    print(f"📁 Analyzing: {latest_wav.name} ({file_size} bytes)")
    
    # Read the raw bytes
    with open(latest_wav, 'rb') as f:
        content = f.read()
    
    print(f"📊 File content (hex): {content[:44].hex()}")
    
    # Try to parse as WAV
    try:
        import wave
        with wave.open(str(latest_wav), 'rb') as wav:
            print(f"📊 WAV Info:")
            print(f"  Channels: {wav.getnchannels()}")
            print(f"  Sample width: {wav.getsampwidth()}")
            print(f"  Frame rate: {wav.getframerate()}")
            print(f"  Frames: {wav.getnframes()}")
            print(f"  Duration: {wav.getnframes() / wav.getframerate():.3f}s")
            
            # Read audio data
            audio_data = wav.readframes(wav.getnframes())
            print(f"  Audio data size: {len(audio_data)} bytes")
            
    except Exception as e:
        print(f"❌ WAV parsing failed: {e}")

def main():
    """Run comprehensive token debugging"""
    print("🔧 Orpheus TTS Token Debug Tool")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    results = {}
    
    # Test 1: LLM token generation
    results['llm_generation'] = test_token_generation() is not None
    
    # Test 2: Token parsing
    results['token_parsing'] = test_token_parsing()
    
    # Test 3: SNAC conversion
    results['snac_conversion'] = test_snac_conversion()
    
    # Test 4: WAV file analysis
    analyze_existing_wav_file()
    
    print("\n" + "=" * 50)
    print("📊 DEBUG RESULTS:")
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test.upper()}: {status}")
    
    failed_tests = [test for test, passed in results.items() if not passed]
    
    if failed_tests:
        print(f"\n⚠️ Failed components: {', '.join(failed_tests)}")
        
        if not results.get('token_parsing'):
            print("\n🔧 LIKELY ISSUE: Token format incompatibility")
            print("   The LLM is generating tokens that can't be converted to audio IDs")
            print("   This suggests the model format doesn't match the expected Orpheus format")
        
        if not results.get('snac_conversion'):
            print("\n🔧 LIKELY ISSUE: SNAC model incompatibility")
            print("   The SNAC decoder can't process the generated tokens")

if __name__ == "__main__":
    main()
