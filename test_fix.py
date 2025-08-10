#!/usr/bin/env python3
"""
Quick test to verify the token fix works
"""

import os
import sys
from pathlib import Path

def test_token_conversion():
    """Test the fixed token conversion"""
    print("🧪 Testing fixed token conversion...")
    
    try:
        # Import the fixed modules
        sys.path.append('.')
        from tts_engine.speechpipe import turn_token_into_id
        
        # Test with the problematic '>' tokens from debug output
        test_tokens = ['>', '>', '>', '>', '>']
        
        print("Testing token conversion:")
        valid_count = 0
        for i, token in enumerate(test_tokens):
            token_id = turn_token_into_id(token, i)
            if token_id is not None and 0 <= token_id <= 4095:
                print(f"  Token '{token}' -> ID {token_id} ✅")
                valid_count += 1
            else:
                print(f"  Token '{token}' -> FAILED ❌")
        
        print(f"\nResult: {valid_count}/{len(test_tokens)} tokens converted successfully")
        return valid_count == len(test_tokens)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run the test"""
    print("🔧 Testing Token Fix")
    print("=" * 30)
    
    os.chdir(Path(__file__).parent)
    
    if test_token_conversion():
        print("\n✅ Token fix is working!")
        print("🚀 You can now restart your TTS server")
        print("\nNext steps:")
        print("1. Stop current TTS server (Ctrl+C)")
        print("2. python app.py --host 0.0.0.0 --port 5005")
        print("3. Test with curl or web interface")
    else:
        print("\n❌ Token fix failed")

if __name__ == "__main__":
    main()
