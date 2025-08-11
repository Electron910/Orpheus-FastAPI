"""
Quick test of production-grade emotion processor
"""

import sys
sys.path.insert(0, '.')

from tts_engine.production_emotion_processor import ProductionEmotionProcessor

def test_production_emotions():
    print("🎭 Testing Production-Grade Emotion Processor")
    print("=" * 50)
    
    processor = ProductionEmotionProcessor()
    
    test_cases = [
        {
            "text": "That joke was absolutely hilarious! I can't stop thinking about it.",
            "voice": "tara",
            "expected": "laugh"
        },
        {
            "text": "Unfortunately, the project didn't go as planned.",
            "voice": "tara", 
            "expected": "sigh"
        },
        {
            "text": "Wow! That's absolutely incredible!",
            "voice": "zoe",
            "expected": "gasp"
        },
        {
            "text": "This is so frustrating and annoying.",
            "voice": "dan",
            "expected": "groan"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test['expected'].upper()} emotion")
        print(f"Original: {test['text']}")
        
        enhanced = processor.process_text(
            test['text'], 
            test['voice'],
            optimize_existing=True,
            add_contextual=True
        )
        
        print(f"Enhanced: {enhanced}")
        
        stats = processor.get_emotion_statistics(enhanced)
        print(f"Stats: {stats['total_emotions']} emotions, "
              f"density: {stats['emotion_density']:.2f}")
        
        if stats['total_emotions'] > 0:
            print("✅ Emotions added successfully")
        else:
            print("⚠️ No emotions detected")
    
    print(f"\n🎯 Production emotion processor test completed!")

if __name__ == "__main__":
    test_production_emotions()
