#!/usr/bin/env python3
"""
Fixed test script for the original Orpheus TTS model
Addresses multiprocessing issues by using proper main guard
"""

def test_orpheus_tts():
    """Test function to run Orpheus TTS with proper error handling"""
    from orpheus_tts import OrpheusModel
    import wave
    import time
    import os
    
    print("🎤 Testing Original Orpheus TTS Model...")
    
    try:
        # Initialize the model
        print("📥 Loading Orpheus model...")
        model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        print("✅ Model loaded successfully!")
        
        prompt = "Hello, this is a test of the original Orpheus TTS system. The weather is nice today."
        
        print(f"🗣️ Generating speech for: '{prompt}'")
        start_time = time.monotonic()
        
        # Generate speech tokens
        syn_tokens = model.generate_speech(
            prompt=prompt,
            voice="tara",
            repetition_penalty=1.1,
            max_tokens=2000,
            temperature=0.4,
            top_p=0.9
        )
        
        # Save to WAV file
        output_file = "original_orpheus_test_fixed.wav"
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
                    print(f"📦 Processed chunk {chunk_count}, frames: {frame_count}")
            
            duration = total_frames / wf.getframerate()
            end_time = time.monotonic()
            
            print(f"🎵 Generated {duration:.2f} seconds of audio in {end_time - start_time:.2f} seconds")
            print(f"📊 Total chunks: {chunk_count}, Total frames: {total_frames}")
            
            # Check file size
            file_size = os.path.getsize(output_file)
            print(f"📁 Output file: {output_file} ({file_size} bytes)")
            
            if file_size > 100:  # More than just WAV header
                print("✅ SUCCESS: Generated valid audio file with original Orpheus TTS!")
                return True
            else:
                print("❌ WARNING: Generated file is too small, may be empty")
                return False
                
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # This is the critical fix for the multiprocessing issue
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    success = test_orpheus_tts()
    if success:
        print("\n🎉 Original Orpheus TTS is working correctly!")
        print("🔧 You can now integrate this into your main application.")
    else:
        print("\n💥 There are still issues with the original Orpheus TTS.")
        print("🔍 Check the error messages above for more details.")
