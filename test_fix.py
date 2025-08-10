#!/usr/bin/env python3
"""
Quick test to verify the TTS fix works
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_token_conversion():
    """Test the fixed token conversion"""
    print("🧪 Testing fixed token conversion...")
    
    try:
        # Import the fixed modules
        sys.path.append('.')
        from tts_engine.speechpipe import turn_token_into_id
        
        # Test with the problematic '>' tokens
        test_tokens = ['>', '>', '>', '>', '>', '>', '>']
        
        print("🔄 Testing token-to-ID conversion:")
        valid_count = 0
        for i, token in enumerate(test_tokens):
            token_id = turn_token_into_id(token, i)
            if token_id is not None:
                valid_count += 1
                print(f"  Token '{token}' at index {i} -> ID {token_id}")
            else:
                print(f"  Token '{token}' at index {i} -> INVALID")
        
        print(f"📊 Valid conversions: {valid_count}/{len(test_tokens)}")
        
        if valid_count == len(test_tokens):
            print("✅ Token conversion fix successful!")
            return True
        else:
            print("❌ Some tokens still invalid")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_generation():
    """Test actual audio generation"""
    print("\n🧪 Testing audio generation...")
    
    try:
        from tts_engine import generate_speech_from_api
        
        # Test with very simple text
        test_text = "Hello"
        output_file = "quick_test.wav"
        
        print(f"🔄 Generating audio for: '{test_text}'")
        
        result = generate_speech_from_api(
            prompt=test_text,
            voice="tara",
            output_file=output_file,
            use_batching=False
        )
        
        # Check result
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 Generated file: {output_file} ({file_size} bytes)")
            
            if file_size > 5000:  # Should be much larger than 44 bytes
                print("✅ Audio generation successful!")
                
                # Quick WAV analysis
                try:
                    import wave
                    with wave.open(output_file, 'rb') as wav:
                        duration = wav.getnframes() / wav.getframerate()
                        print(f"🎵 Duration: {duration:.2f} seconds")
                        print(f"🎵 Sample rate: {wav.getframerate()} Hz")
                except:
                    pass
                
                return True
            else:
                print(f"❌ File still too small: {file_size} bytes")
                return False
        else:
            print("❌ No output file generated")
            return False
            
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick tests"""
    print("🔧 Quick TTS Fix Verification")
    print("=" * 40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Test 1: Token conversion
    token_test = test_token_conversion()
    
    # Test 2: Audio generation
    audio_test = test_audio_generation()
    
    print("\n" + "=" * 40)
    print("📊 TEST RESULTS:")
    print(f"  Token Conversion: {'✅ PASS' if token_test else '❌ FAIL'}")
    print(f"  Audio Generation: {'✅ PASS' if audio_test else '❌ FAIL'}")
    
    if token_test and audio_test:
        print("\n🎉 TTS FIX SUCCESSFUL!")
        print("🚀 Your TTS system should now work properly")
        print("\n📋 Next steps:")
        print("1. Restart TTS server: python app.py --host 0.0.0.0 --port 5005")
        print("2. Test with your voice agent")
    else:
        print("\n⚠️ Some tests failed - check the errors above")

if __name__ == "__main__":
    main()
