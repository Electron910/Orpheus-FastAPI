"""
Comprehensive validation of production-grade emotion processing system
"""

import sys
import os
sys.path.insert(0, '.')

def test_imports():
    """Test all imports are working correctly"""
    print("🔧 Testing Production System Imports")
    print("=" * 50)
    
    try:
        from tts_engine.production_emotion_processor import ProductionEmotionProcessor
        print("✅ ProductionEmotionProcessor imported successfully")
        
        from tts_engine.production_emotion_processor import production_emotion_processor
        print("✅ Global production_emotion_processor imported successfully")
        
        from tts_engine import production_emotion_processor as pkg_processor
        print("✅ Package-level production_emotion_processor imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_emotion_processing():
    """Test emotion processing functionality"""
    print("\n🎭 Testing Emotion Processing Functionality")
    print("=" * 50)
    
    try:
        from tts_engine.production_emotion_processor import ProductionEmotionProcessor
        processor = ProductionEmotionProcessor()
        
        # Test cases with expected behaviors
        test_cases = [
            {
                "name": "Laughter Context",
                "text": "That joke was absolutely hilarious! I can't stop thinking about it.",
                "voice": "tara",
                "should_add_emotion": True
            },
            {
                "name": "Disappointment Context", 
                "text": "Unfortunately, the project didn't go as planned. I'm quite disappointed.",
                "voice": "tara",
                "should_add_emotion": True
            },
            {
                "name": "Surprise Context",
                "text": "Wow! That's absolutely incredible! I never expected this result.",
                "voice": "zoe", 
                "should_add_emotion": True
            },
            {
                "name": "Neutral Text",
                "text": "The weather is nice today.",
                "voice": "tara",
                "should_add_emotion": False
            }
        ]
        
        success_count = 0
        
        for test in test_cases:
            print(f"\n📝 {test['name']}")
            print(f"   Input: {test['text']}")
            
            # Process text
            enhanced = processor.process_text(
                test['text'],
                test['voice'],
                optimize_existing=True,
                add_contextual=True,
                max_contextual_additions=2
            )
            
            print(f"   Output: {enhanced}")
            
            # Get statistics
            stats = processor.get_emotion_statistics(enhanced)
            emotion_count = stats['total_emotions']
            
            print(f"   Emotions: {emotion_count}, Density: {stats['emotion_density']:.2f}")
            
            # Validate expectation
            if test['should_add_emotion'] and emotion_count > 0:
                print("   ✅ SUCCESS: Emotions added as expected")
                success_count += 1
            elif not test['should_add_emotion'] and emotion_count == 0:
                print("   ✅ SUCCESS: No emotions added as expected")
                success_count += 1
            else:
                print(f"   ⚠️ UNEXPECTED: Expected emotions={test['should_add_emotion']}, got {emotion_count}")
        
        print(f"\n🎯 Emotion Processing Results: {success_count}/{len(test_cases)} tests passed")
        return success_count == len(test_cases)
        
    except Exception as e:
        print(f"❌ Emotion processing test error: {e}")
        return False

def test_voice_compatibility():
    """Test voice-emotion compatibility system"""
    print("\n🎤 Testing Voice-Emotion Compatibility")
    print("=" * 50)
    
    try:
        from tts_engine.production_emotion_processor import ProductionEmotionProcessor
        processor = ProductionEmotionProcessor()
        
        # Test voice-specific optimizations
        voice_tests = [
            {"voice": "tara", "emotion": "laugh", "expected_high_compatibility": True},
            {"voice": "dan", "emotion": "groan", "expected_high_compatibility": True},
            {"voice": "zoe", "emotion": "gasp", "expected_high_compatibility": True},
            {"voice": "leo", "emotion": "chuckle", "expected_high_compatibility": True}
        ]
        
        success_count = 0
        
        for test in voice_tests:
            voice = test['voice']
            emotion = test['emotion']
            
            if emotion in processor.emotion_tags:
                compatibility = processor.emotion_tags[emotion].voice_compatibility.get(voice, 0.5)
                print(f"   {voice} + {emotion}: {compatibility:.2f} compatibility")
                
                if test['expected_high_compatibility'] and compatibility >= 0.8:
                    print("   ✅ High compatibility as expected")
                    success_count += 1
                elif not test['expected_high_compatibility'] and compatibility < 0.8:
                    print("   ✅ Low compatibility as expected")
                    success_count += 1
                else:
                    print(f"   ⚠️ Unexpected compatibility level")
            else:
                print(f"   ❌ Emotion {emotion} not found")
        
        print(f"\n🎯 Voice Compatibility Results: {success_count}/{len(voice_tests)} tests passed")
        return success_count == len(voice_tests)
        
    except Exception as e:
        print(f"❌ Voice compatibility test error: {e}")
        return False

def test_context_analysis():
    """Test context analysis functionality"""
    print("\n🧠 Testing Context Analysis")
    print("=" * 50)
    
    try:
        from tts_engine.production_emotion_processor import ProductionEmotionProcessor
        processor = ProductionEmotionProcessor()
        
        context_tests = [
            {
                "text": "That's absolutely amazing and incredible!",
                "expected_contexts": ["high_excitement"]
            },
            {
                "text": "Unfortunately, this is quite disappointing.",
                "expected_contexts": ["negative_context"]
            },
            {
                "text": "What do you think about this? Really?",
                "expected_contexts": ["question_context"]
            }
        ]
        
        success_count = 0
        
        for test in context_tests:
            print(f"   Text: {test['text']}")
            context_scores = processor.analyze_text_context(test['text'])
            print(f"   Detected contexts: {list(context_scores.keys())}")
            
            # Check if expected contexts are detected
            found_expected = any(ctx in context_scores for ctx in test['expected_contexts'])
            if found_expected:
                print("   ✅ Expected context detected")
                success_count += 1
            else:
                print("   ⚠️ Expected context not detected")
        
        print(f"\n🎯 Context Analysis Results: {success_count}/{len(context_tests)} tests passed")
        return success_count == len(context_tests)
        
    except Exception as e:
        print(f"❌ Context analysis test error: {e}")
        return False

def main():
    """Run comprehensive validation"""
    print("🎭 ORPHEUS TTS PRODUCTION-GRADE EMOTION SYSTEM VALIDATION")
    print("=" * 70)
    
    # Run all tests
    tests = [
        ("Import System", test_imports),
        ("Emotion Processing", test_emotion_processing),
        ("Voice Compatibility", test_voice_compatibility), 
        ("Context Analysis", test_context_analysis)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        if test_func():
            passed_tests += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n🏁 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL PRODUCTION SYSTEM TESTS PASSED!")
        print("\n✨ Production-Grade Emotion Processing System is ready!")
        print("\n📋 Key Improvements:")
        print("   • Context-aware emotion detection and placement")
        print("   • Voice-specific emotion compatibility optimization")
        print("   • Intelligent emotion intensity adjustment")
        print("   • Production-grade error handling and fallbacks")
        print("   • Comprehensive emotion statistics and monitoring")
        
        print("\n🚀 Next Steps:")
        print("   1. Start TTS server to test integrated system")
        print("   2. Run production emotion tests with audio generation")
        print("   3. Compare audio quality with previous implementation")
        print("   4. Deploy to production environment")
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed - system needs attention")

if __name__ == "__main__":
    main()
