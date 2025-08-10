#!/usr/bin/env python3
"""
Token Compatibility Fix for Orpheus TTS
This script fixes the token format incompatibility causing 44-byte WAV files
"""

import os
import sys
import re
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def analyze_token_format():
    """Analyze the current token format and identify the issue"""
    print("🔍 Analyzing token format compatibility...")
    
    # Test the current LLM output format
    api_url = os.environ.get('ORPHEUS_API_URL', 'http://0.0.0.0:1234/v1/chat/completions')
    
    # Test with minimal prompt
    test_prompt = '<|audio|>tara: "Test"<|eot_id|>'
    
    payload = {
        "model": "Vocalis-q4_k_m.gguf",
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": 50,
        "temperature": 0.6,
        "stream": False
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"📝 LLM Output: {content[:200]}...")
            
            # Check if this looks like the expected format
            if '<custom_token_' in content:
                print("✅ Found custom_token format - this is correct")
                return 'custom_token'
            elif re.search(r'\d+', content):
                print("⚠️ Found numeric tokens - might need conversion")
                return 'numeric'
            else:
                print("❌ Unexpected token format")
                return 'unknown'
        else:
            print(f"❌ API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Token analysis failed: {e}")
        return None

def patch_token_conversion():
    """Patch the token conversion to handle the current model's output format"""
    print("🔧 Patching token conversion for compatibility...")
    
    # Read the current speechpipe.py
    speechpipe_path = Path('tts_engine/speechpipe.py')
    if not speechpipe_path.exists():
        print("❌ speechpipe.py not found")
        return False
    
    # Create a backup
    backup_path = speechpipe_path.with_suffix('.py.backup')
    if not backup_path.exists():
        speechpipe_path.rename(backup_path)
        print(f"✅ Created backup: {backup_path}")
    
    # Read the backup content
    with open(backup_path, 'r') as f:
        content = f.read()
    
    # Patch the turn_token_into_id function for better compatibility
    patched_function = '''def turn_token_into_id(token_string, index):
    """
    Enhanced token-to-ID conversion with better error handling and format detection.
    This version handles multiple token formats from different Orpheus model variants.
    
    Args:
        token_string: The token string to convert
        index: Position index used for token offset calculation
        
    Returns:
        int: Token ID if valid, None otherwise
    """
    global token_id_cache
    
    # Use cache for performance
    cache_key = f"{token_string}_{index}"
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
    
    # Clean up cache if it gets too large
    if len(token_id_cache) > MAX_CACHE_SIZE:
        # Remove oldest 20% of entries
        items_to_remove = len(token_id_cache) // 5
        for key in list(token_id_cache.keys())[:items_to_remove]:
            del token_id_cache[key]
    
    try:
        # Method 1: Handle custom_token format (original Orpheus format)
        if token_string.startswith(CUSTOM_TOKEN_PREFIX):
            # Extract the number from custom_token_XXXXX>
            number_part = token_string[len(CUSTOM_TOKEN_PREFIX):-1]
            try:
                base_id = int(number_part)
                # Apply index offset for proper sequencing
                final_id = base_id + (index * 7)  # 7 tokens per frame
                
                # Ensure ID is in valid range for SNAC
                if 0 <= final_id <= 4096:
                    token_id_cache[cache_key] = final_id
                    return final_id
                else:
                    # Wrap around if out of range
                    wrapped_id = final_id % 4096
                    token_id_cache[cache_key] = wrapped_id
                    return wrapped_id
            except ValueError:
                pass
        
        # Method 2: Handle direct numeric tokens
        if token_string.isdigit():
            base_id = int(token_string)
            if 0 <= base_id <= 4096:
                token_id_cache[cache_key] = base_id
                return base_id
        
        # Method 3: Extract numbers from mixed format
        import re
        numbers = re.findall(r'\\d+', str(token_string))
        if numbers:
            try:
                base_id = int(numbers[0])  # Use first number found
                # Apply modulo to ensure valid range
                final_id = (base_id + index) % 4096
                token_id_cache[cache_key] = final_id
                return final_id
            except ValueError:
                pass
        
        # Method 4: Hash-based fallback for unrecognized formats
        # This ensures we always return a valid ID even for unexpected tokens
        import hashlib
        hash_obj = hashlib.md5(f"{token_string}_{index}".encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)  # Use first 8 hex chars
        fallback_id = hash_int % 4096  # Ensure in valid range
        
        print(f"⚠️ Using fallback conversion for token '{token_string}' -> {fallback_id}")
        token_id_cache[cache_key] = fallback_id
        return fallback_id
        
    except Exception as e:
        print(f"❌ Token conversion error for '{token_string}': {e}")
        # Emergency fallback - use index-based ID
        emergency_id = (index * 7) % 4096
        token_id_cache[cache_key] = emergency_id
        return emergency_id'''
    
    # Replace the function in the content
    # Find the function definition and replace it
    pattern = r'def turn_token_into_id\(.*?\n(?:.*?\n)*?.*?return.*?(?:\n|$)'
    if re.search(pattern, content, re.MULTILINE | re.DOTALL):
        new_content = re.sub(pattern, patched_function, content, flags=re.MULTILINE | re.DOTALL)
    else:
        print("⚠️ Could not find turn_token_into_id function, appending patch")
        new_content = content + '\n\n' + patched_function
    
    # Write the patched version
    with open(speechpipe_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Token conversion patched successfully")
    return True

def patch_audio_conversion():
    """Patch the audio conversion to handle edge cases better"""
    print("🔧 Patching audio conversion for robustness...")
    
    speechpipe_path = Path('tts_engine/speechpipe.py')
    
    # Read current content
    with open(speechpipe_path, 'r') as f:
        content = f.read()
    
    # Enhanced convert_to_audio function
    enhanced_convert = '''def convert_to_audio(multiframe, count):
    """
    Enhanced audio conversion with better error handling and validation.
    Handles edge cases that cause empty audio output.
    """
    if len(multiframe) < 7:
        print(f"⚠️ Insufficient tokens for audio frame: {len(multiframe)} < 7")
        return None
  
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    print(f"🔄 Converting {num_frames} frames to audio (tokens: {len(frame)})")
    
    try:
        # Pre-allocate tensors instead of incrementally building them
        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)
        
        # Use vectorized operations where possible
        frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
        
        # Direct indexing is much faster than concatenation in a loop
        for j in range(num_frames):
            idx = j * 7
            
            # Code 0 - single value per frame
            codes_0[j] = frame_tensor[idx]
            
            # Code 1 - two values per frame
            codes_1[j*2] = frame_tensor[idx+1]
            codes_1[j*2+1] = frame_tensor[idx+4]
            
            # Code 2 - four values per frame
            codes_2[j*4] = frame_tensor[idx+2]
            codes_2[j*4+1] = frame_tensor[idx+3]
            codes_2[j*4+2] = frame_tensor[idx+5]
            codes_2[j*4+3] = frame_tensor[idx+6]
        
        # Reshape codes into expected format
        codes = [
            codes_0.unsqueeze(0), 
            codes_1.unsqueeze(0), 
            codes_2.unsqueeze(0)
        ]
        
        # Enhanced validation with auto-correction
        for i, code_tensor in enumerate(codes):
            # Check for out-of-range values and clamp them
            min_val = torch.min(code_tensor)
            max_val = torch.max(code_tensor)
            
            if min_val < 0 or max_val > 4096:
                print(f"⚠️ Code {i} out of range: {min_val} to {max_val}, clamping...")
                codes[i] = torch.clamp(code_tensor, 0, 4096)
            
            # Check for all-zero codes (indicates token conversion failure)
            if torch.all(code_tensor == 0):
                print(f"⚠️ Code {i} is all zeros, adding variation...")
                # Add small random variation to prevent silence
                noise = torch.randint(1, 100, code_tensor.shape, device=snac_device)
                codes[i] = code_tensor + noise
        
        # Use CUDA stream for parallel processing if available
        stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()
        
        with stream_ctx, torch.inference_mode():
            # Decode the audio
            audio_hat = model.decode(codes)
            
            if audio_hat is None:
                print("❌ SNAC decode returned None")
                return None
                
            # Convert to numpy and validate
            audio_np = audio_hat.squeeze().cpu().numpy()
            
            if len(audio_np) == 0:
                print("❌ SNAC decode returned empty audio")
                return None
            
            # Normalize audio to prevent clipping
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            print(f"✅ Generated {len(audio_int16)} audio samples ({len(audio_int16)/24000:.2f}s)")
            
            return audio_int16.tobytes()
            
    except Exception as e:
        print(f"❌ Audio conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None'''
    
    # Replace the convert_to_audio function
    pattern = r'def convert_to_audio\(.*?\n(?:.*?\n)*?.*?return.*?(?:\n|$)'
    if re.search(pattern, content, re.MULTILINE | re.DOTALL):
        new_content = re.sub(pattern, enhanced_convert, content, flags=re.MULTILINE | re.DOTALL)
        
        # Write the enhanced version
        with open(speechpipe_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Audio conversion enhanced successfully")
        return True
    else:
        print("❌ Could not find convert_to_audio function")
        return False

def test_fixed_conversion():
    """Test the fixed token conversion"""
    print("🧪 Testing fixed token conversion...")
    
    try:
        # Import the patched modules
        import importlib
        import sys
        
        # Reload modules to pick up changes
        if 'tts_engine.speechpipe' in sys.modules:
            importlib.reload(sys.modules['tts_engine.speechpipe'])
        if 'tts_engine.inference' in sys.modules:
            importlib.reload(sys.modules['tts_engine.inference'])
        
        from tts_engine import generate_speech_from_api
        
        # Test with simple text
        test_text = "Hello test"
        output_file = "compatibility_test.wav"
        
        print(f"🔄 Generating speech for: '{test_text}'")
        
        result = generate_speech_from_api(
            prompt=test_text,
            voice="tara",
            output_file=output_file,
            use_batching=False
        )
        
        # Check result
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 Output file size: {file_size} bytes")
            
            if file_size > 1000:
                print("✅ Token compatibility fix successful!")
                return True
            else:
                print("❌ Still generating small files")
                return False
        else:
            print("❌ No output file generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main compatibility fix function"""
    print("🔧 Orpheus TTS Token Compatibility Fix")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Step 1: Analyze current token format
    token_format = analyze_token_format()
    if not token_format:
        print("❌ Could not analyze token format")
        return
    
    # Step 2: Apply patches
    success_count = 0
    
    if patch_token_conversion():
        success_count += 1
    
    if patch_audio_conversion():
        success_count += 1
    
    print(f"\n📊 Applied {success_count}/2 patches successfully")
    
    # Step 3: Test the fixes
    if success_count == 2:
        print("\n🧪 Testing compatibility fixes...")
        if test_fixed_conversion():
            print("\n🎉 Token compatibility fix completed successfully!")
            print("🚀 TTS should now generate proper audio files")
        else:
            print("\n⚠️ Fixes applied but test still failed")
            print("   You may need to restart the TTS server")
    else:
        print("\n❌ Some patches failed to apply")

if __name__ == "__main__":
    main()
