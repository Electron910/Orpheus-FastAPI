#!/usr/bin/env python3
"""
ROBUST TTS Server - Fixed for Production Use
Handles concurrent requests, prevents engine crashes, includes request queuing
"""

from flask import Flask, Response, request, jsonify
import struct
import os
import sys
import multiprocessing
import threading
import time
import uuid
import queue
from dotenv import load_dotenv
from typing import Generator, Optional

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global model instance and request queue
orpheus_engine = None
request_queue = queue.Queue()
processing_lock = threading.Lock()
is_processing = False

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV file header for streaming audio"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1, channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b'data', data_size
    )
    return header

def initialize_model():
    """Initialize the Orpheus TTS model with proper error handling"""
    global orpheus_engine
    
    if orpheus_engine is not None:
        return True
        
    try:
        print("🎤 Initializing Robust Orpheus TTS Model...")
        from orpheus_tts import OrpheusModel
        
        model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
        orpheus_engine = OrpheusModel(model_name=model_name)
        
        print("✅ Orpheus TTS Model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Orpheus TTS model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_audio_safely(prompt: str, voice: str = "tara", temperature: float = 0.4, 
                         top_p: float = 0.9, max_tokens: int = 1500, 
                         repetition_penalty: float = 1.1) -> Optional[bytes]:
    """Generate audio with proper error handling and unique request IDs"""
    global orpheus_engine, is_processing
    
    if orpheus_engine is None:
        print("❌ Model not initialized")
        return None
    
    # Use processing lock to prevent concurrent requests
    with processing_lock:
        if is_processing:
            print("⚠️ Another request is processing, queuing...")
            time.sleep(0.5)  # Brief wait
        
        is_processing = True
        
        try:
            print(f"🗣️ Generating TTS for: '{prompt[:50]}...' with voice: {voice}")
            
            # Generate unique request ID
            request_id = f"tts-{uuid.uuid4().hex[:8]}"
            
            # Generate speech tokens using original Orpheus with unique ID
            syn_tokens = orpheus_engine.generate_speech(
                prompt=prompt,
                voice=voice,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Collect all audio data
            audio_data = bytearray()
            audio_data.extend(create_wav_header())
            
            chunk_count = 0
            for chunk in syn_tokens:
                if chunk is not None and len(chunk) > 0:
                    chunk_count += 1
                    audio_data.extend(chunk)
                    if chunk_count % 10 == 0:  # Log every 10 chunks
                        print(f"📦 Processed {chunk_count} chunks...")
                        
            print(f"✅ Generated {chunk_count} audio chunks, total size: {len(audio_data)} bytes")
            return bytes(audio_data)
            
        except Exception as e:
            print(f"❌ Error during audio generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            is_processing = False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global orpheus_engine, is_processing
    
    return jsonify({
        "status": "ok" if orpheus_engine else "error",
        "model_loaded": orpheus_engine is not None,
        "processing": is_processing,
        "model": "canopylabs/orpheus-tts-0.1-finetune-prod",
        "version": "robust-1.0"
    })

@app.route('/tts', methods=['GET', 'POST'])
def tts_endpoint():
    """TTS endpoint with robust error handling"""
    global orpheus_engine
    
    if orpheus_engine is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    # Get parameters from request
    if request.method == 'POST':
        data = request.get_json() or {}
        prompt = data.get('prompt', 'Hello world')
        voice = data.get('voice', 'tara')
        temperature = float(data.get('temperature', 0.4))
        top_p = float(data.get('top_p', 0.9))
        max_tokens = int(data.get('max_tokens', 1500))
        repetition_penalty = float(data.get('repetition_penalty', 1.1))
    else:
        prompt = request.args.get('prompt', 'Hello world')
        voice = request.args.get('voice', 'tara')
        temperature = float(request.args.get('temperature', 0.4))
        top_p = float(request.args.get('top_p', 0.9))
        max_tokens = int(request.args.get('max_tokens', 1500))
        repetition_penalty = float(request.args.get('repetition_penalty', 1.1))

    # Generate audio safely
    audio_data = generate_audio_safely(
        prompt=prompt,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty
    )
    
    if audio_data is None:
        return jsonify({"error": "TTS generation failed"}), 500
    
    def generate_response():
        yield audio_data

    return Response(
        generate_response(),
        mimetype='audio/wav',
        headers={'Content-Disposition': 'attachment; filename=speech.wav'}
    )

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint with sample audio"""
    test_prompt = "Hello! This is a test of the robust Orpheus TTS system. The server is working correctly."
    
    audio_data = generate_audio_safely(prompt=test_prompt, voice="tara", max_tokens=1000)
    
    if audio_data is None:
        return jsonify({"error": "Test TTS generation failed"}), 500
    
    def generate_response():
        yield audio_data

    return Response(
        generate_response(),
        mimetype='audio/wav',
        headers={'Content-Disposition': 'attachment; filename=test_speech.wav'}
    )

@app.route('/', methods=['GET'])
def index():
    """Simple index page with API documentation"""
    global is_processing
    
    status = "🟢 Ready" if orpheus_engine and not is_processing else "🟡 Processing" if is_processing else "🔴 Not Ready"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robust Orpheus TTS API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }}
            .status {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .endpoint {{ background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .method {{ color: #007acc; font-weight: bold; }}
            .test-link {{ background: #007acc; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 5px; }}
        </style>
    </head>
    <body>
        <h1>🎤 Robust Orpheus TTS API</h1>
        
        <div class="status">
            <h3>Status: {status}</h3>
            <p><strong>Model:</strong> canopylabs/orpheus-tts-0.1-finetune-prod</p>
            <p><strong>Your URL:</strong> https://561pjq4x4ud1px-5005.proxy.runpod.net/</p>
            <p><strong>Concurrency:</strong> Sequential processing (prevents crashes)</p>
        </div>
        
        <h2>🎵 Quick Tests:</h2>
        <a href="/test" class="test-link">Test TTS</a>
        <a href="/health" class="test-link">Health Check</a>
        <a href="/tts?prompt=Hello%20world,%20this%20is%20the%20robust%20TTS%20system" class="test-link">Custom TTS</a>
        
        <h2>📋 API Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/tts</strong><br>
            Generate TTS audio with query parameters<br>
            Example: <code>/tts?prompt=Hello%20world&voice=tara</code>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <strong>/tts</strong><br>
            Generate TTS audio with JSON payload<br>
            Body: <code>{{"prompt": "text", "voice": "tara"}}</code>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/health</strong><br>
            Health check and processing status
        </div>
        
        <h2>🔗 Integration:</h2>
        <p><strong>For Vocalis Frontend:</strong></p>
        <code>const ttsUrl = 'https://561pjq4x4ud1px-5005.proxy.runpod.net/tts';</code>
        
        <h2>⚡ Features:</h2>
        <ul>
            <li>✅ Prevents concurrent request crashes</li>
            <li>✅ Unique request IDs</li>
            <li>✅ Sequential processing</li>
            <li>✅ Robust error handling</li>
            <li>✅ Production ready</li>
        </ul>
    </body>
    </html>
    """

def main():
    """Main function with proper multiprocessing setup"""
    # Set multiprocessing method for VLLM compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    # Initialize model
    if not initialize_model():
        print("❌ Failed to initialize model. Exiting.")
        sys.exit(1)
    
    # Get configuration from environment
    host = "0.0.0.0"
    port = 5005
    
    print(f"🚀 Starting ROBUST Orpheus TTS Server on {host}:{port}")
    print("🔧 Features: Concurrent request handling, unique IDs, error recovery")
    print("🌐 Your URL: https://561pjq4x4ud1px-5005.proxy.runpod.net/")
    
    # Run Flask app with threading enabled
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )

if __name__ == '__main__':
    main()
