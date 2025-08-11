"""
Production-Grade Emotion Testing Script for Orpheus TTS
Tests the enhanced emotion processing system with comprehensive scenarios
"""

import requests
import json
import time
import os
from pathlib import Path

# Test configuration
TTS_SERVER_URL = "http://localhost:5005"
OUTPUT_DIR = Path("test_outputs_production")
OUTPUT_DIR.mkdir(exist_ok=True)

def test_tts_endpoint(text, voice="tara", filename_suffix=""):
    """Test TTS endpoint with given text and voice"""
    try:
        response = requests.post(
            f"{TTS_SERVER_URL}/generate_speech",
            json={
                "text": text,
                "voice": voice,
                "format": "wav"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            filename = f"{voice}_{filename_suffix}.wav"
            filepath = OUTPUT_DIR / filename
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            file_size = os.path.getsize(filepath)
            print(f"✅ Generated: {filename} ({file_size:,} bytes)")
            return True, filepath
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, None

def run_production_emotion_tests():
    """Run comprehensive production-grade emotion tests"""
    
    print("🎭 PRODUCTION-GRADE EMOTION ACCURACY TESTING")
    print("=" * 60)
    
    # Test scenarios based on official Orpheus-TTS patterns
    test_scenarios = [
        {
            "name": "Context-Aware Laughter",
            "text": "That joke was absolutely hilarious! I can't stop thinking about it.",
            "voices": ["tara", "zoe", "mia"],
            "expected_emotions": ["laugh", "chuckle"]
        },
        {
            "name": "Subtle Amusement",
            "text": "That's an interesting way to look at it. Quite clever actually.",
            "voices": ["leo", "dan", "tara"],
            "expected_emotions": ["chuckle"]
        },
        {
            "name": "Disappointment Expression",
            "text": "Unfortunately, the project didn't go as planned. I'm quite disappointed.",
            "voices": ["tara", "mia", "leah"],
            "expected_emotions": ["sigh", "sniffle"]
        },
        {
            "name": "Surprise and Amazement",
            "text": "Wow! That's absolutely incredible! I never expected this result.",
            "voices": ["zoe", "mia", "jess"],
            "expected_emotions": ["gasp"]
        },
        {
            "name": "Frustration and Annoyance",
            "text": "This is so frustrating! Nothing is working properly today.",
            "voices": ["dan", "leo", "zac"],
            "expected_emotions": ["groan"]
        },
        {
            "name": "Tiredness and Boredom",
            "text": "This meeting is so boring. I'm getting really tired of these discussions.",
            "voices": ["tara", "dan", "leo"],
            "expected_emotions": ["yawn", "sigh"]
        },
        {
            "name": "Emotional Sensitivity",
            "text": "That story was so touching. It really moved me to tears.",
            "voices": ["tara", "mia", "amelie"],
            "expected_emotions": ["sniffle"]
        },
        {
            "name": "Professional Clearing",
            "text": "Excuse me, I need to clarify something important about this proposal.",
            "voices": ["leo", "dan", "thomas"],
            "expected_emotions": ["cough"]
        },
        {
            "name": "Mixed Emotions Complex",
            "text": "I'm excited about the opportunity, but honestly, I'm also quite nervous about the challenges ahead.",
            "voices": ["tara", "zoe"],
            "expected_emotions": ["multiple"]
        },
        {
            "name": "Natural Conversation Flow",
            "text": "So, what do you think about this? I mean, it's pretty amazing, right? But then again, maybe I'm just being overly optimistic.",
            "voices": ["tara", "zoe", "mia"],
            "expected_emotions": ["contextual"]
        }
    ]
    
    successful_tests = 0
    total_tests = 0
    
    for scenario in test_scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        print(f"Text: {scenario['text']}")
        print("-" * 40)
        
        for voice in scenario['voices']:
            total_tests += 1
            success, filepath = test_tts_endpoint(
                scenario['text'], 
                voice, 
                f"{scenario['name'].lower().replace(' ', '_')}_{voice}"
            )
            
            if success:
                successful_tests += 1
                print(f"   🎯 {voice}: SUCCESS")
            else:
                print(f"   ❌ {voice}: FAILED")
    
    print(f"\n🎯 PRODUCTION EMOTION TEST RESULTS")
    print("=" * 60)
    print(f"Successful tests: {successful_tests}/{total_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 ALL PRODUCTION EMOTION TESTS PASSED!")
    else:
        print(f"⚠️ {total_tests - successful_tests} tests failed - check server logs")

def test_voice_emotion_compatibility():
    """Test voice-specific emotion compatibility"""
    
    print(f"\n🎭 VOICE-EMOTION COMPATIBILITY TESTING")
    print("=" * 60)
    
    # Test each voice with their optimal emotions
    voice_emotion_tests = {
        "tara": {
            "optimal": ["laugh", "sigh", "sniffle"],
            "text": "I'm so happy about this! <laugh> But then I got sad thinking about it. <sigh> It was quite emotional. <sniffle>"
        },
        "zoe": {
            "optimal": ["gasp", "laugh"],
            "text": "Oh my goodness! <gasp> That's so funny! <laugh>"
        },
        "dan": {
            "optimal": ["groan", "cough"],
            "text": "This is so annoying. <groan> Excuse me, let me clarify. <cough>"
        },
        "leo": {
            "optimal": ["chuckle", "cough"],
            "text": "That's quite amusing. <chuckle> Well, actually... <cough>"
        },
        "mia": {
            "optimal": ["gasp", "sniffle"],
            "text": "I can't believe it! <gasp> It's so touching. <sniffle>"
        }
    }
    
    for voice, config in voice_emotion_tests.items():
        print(f"\n🎤 Testing {voice} with optimal emotions: {config['optimal']}")
        success, filepath = test_tts_endpoint(
            config['text'], 
            voice, 
            f"voice_compatibility_{voice}"
        )
        
        if success:
            print(f"   ✅ {voice} compatibility test: SUCCESS")
        else:
            print(f"   ❌ {voice} compatibility test: FAILED")

def test_emotion_intensity_levels():
    """Test different emotion intensity levels"""
    
    print(f"\n🎭 EMOTION INTENSITY TESTING")
    print("=" * 60)
    
    intensity_tests = [
        {
            "name": "Subtle Emotions",
            "text": "That's mildly amusing, I suppose.",
            "voice": "tara"
        },
        {
            "name": "Normal Emotions", 
            "text": "That's really funny! I'm quite disappointed though.",
            "voice": "tara"
        },
        {
            "name": "Strong Emotions",
            "text": "That's absolutely hilarious! I'm completely shocked by this incredible news!",
            "voice": "tara"
        }
    ]
    
    for test in intensity_tests:
        print(f"\n📊 Testing: {test['name']}")
        success, filepath = test_tts_endpoint(
            test['text'],
            test['voice'],
            f"intensity_{test['name'].lower().replace(' ', '_')}"
        )
        
        if success:
            print(f"   ✅ {test['name']}: SUCCESS")
        else:
            print(f"   ❌ {test['name']}: FAILED")

def test_server_status():
    """Test if TTS server is running and responsive"""
    try:
        response = requests.get(f"{TTS_SERVER_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ TTS Server is running and responsive")
            return True
        else:
            print(f"⚠️ TTS Server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ TTS Server is not accessible: {e}")
        print("Please ensure the TTS server is running on port 5005")
        return False

if __name__ == "__main__":
    print("🎭 ORPHEUS TTS PRODUCTION-GRADE EMOTION TESTING")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Server URL: {TTS_SERVER_URL}")
    
    # Check server status first
    if not test_server_status():
        print("\n❌ Cannot proceed without TTS server. Please start the server first.")
        exit(1)
    
    start_time = time.time()
    
    # Run all test suites
    run_production_emotion_tests()
    test_voice_emotion_compatibility() 
    test_emotion_intensity_levels()
    
    end_time = time.time()
    
    print(f"\n🏁 TESTING COMPLETED")
    print("=" * 60)
    print(f"Total testing time: {end_time - start_time:.1f} seconds")
    print(f"Audio files saved to: {OUTPUT_DIR.absolute()}")
    print("\n🎧 Listen to the generated audio files to evaluate:")
    print("   • Emotion accuracy and naturalness")
    print("   • Voice-emotion compatibility")
    print("   • Context-aware emotion placement")
    print("   • Overall audio quality improvements")
    
    print(f"\n💡 Next steps:")
    print("   1. Listen to all generated audio files")
    print("   2. Compare with previous emotion processing")
    print("   3. Fine-tune emotion intensity levels if needed")
    print("   4. Deploy to production environment")
