#!/usr/bin/env python3
"""
Streaming TTS Server using Original Orpheus TTS Model
Based on the Flask streaming example but with better error handling
"""

from flask import Flask, Response, request, jsonify
import struct
import os
import sys
import multiprocessing
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global model instance
engine = None

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV file header for streaming audio"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0  # Unknown size for streaming

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,                   # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

def initialize_model():
    """Initialize the Orpheus TTS model with proper error handling"""
    global engine
    
    if engine is not None:
        return True
        
    try:
        print("🎤 Initializing Original Orpheus TTS Model...")
        from orpheus_tts import OrpheusModel
        
        model_name = os.environ.get('ORPHEUS_MODEL_NAME', 'canopylabs/orpheus-tts-0.1-finetune-prod')
        engine = OrpheusModel(model_name=model_name)
        
        print("✅ Orpheus TTS Model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Orpheus TTS model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if engine is None:
        return jsonify({"status": "error", "message": "Model not initialized"}), 500
    return jsonify({"status": "ok", "model": "orpheus-tts"})

@app.route('/tts', methods=['GET', 'POST'])
def tts_endpoint():
    """TTS endpoint with streaming audio response"""
    if engine is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    # Get parameters from request
    if request.method == 'POST':
        data = request.get_json() or {}
        prompt = data.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
        voice = data.get('voice', 'tara')
        temperature = float(data.get('temperature', 0.4))
        top_p = float(data.get('top_p', 0.9))
        max_tokens = int(data.get('max_tokens', 2000))
        repetition_penalty = float(data.get('repetition_penalty', 1.1))
    else:
        prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
        voice = request.args.get('voice', 'tara')
        temperature = float(request.args.get('temperature', 0.4))
        top_p = float(request.args.get('top_p', 0.9))
        max_tokens = int(request.args.get('max_tokens', 2000))
        repetition_penalty = float(request.args.get('repetition_penalty', 1.1))

    print(f"🗣️ Generating TTS for: '{prompt[:50]}...' with voice: {voice}")

    def generate_audio_stream():
        """Generator function for streaming audio"""
        try:
            # Send WAV header first
            yield create_wav_header()

            # Generate speech tokens
            syn_tokens = engine.generate_speech(
                prompt=prompt,
                voice=voice,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            chunk_count = 0
            for chunk in syn_tokens:
                if chunk is not None and len(chunk) > 0:
                    chunk_count += 1
                    print(f"📦 Streaming chunk {chunk_count}, size: {len(chunk)} bytes")
                    yield chunk
                    
            print(f"✅ Finished streaming {chunk_count} audio chunks")
            
        except Exception as e:
            print(f"❌ Error during audio generation: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty audio on error
            yield b''

    return Response(generate_audio_stream(), mimetype='audio/wav')

@app.route('/tts_test', methods=['GET'])
def tts_test():
    """Test endpoint for quick TTS validation"""
    test_prompts = [
        "Hello, this is a test of the streaming TTS system.",
        "The weather is beautiful today.",
        "Testing one, two, three."
    ]
    
    prompt = request.args.get('prompt') or test_prompts[0]
    
    return tts_endpoint()

@app.route('/', methods=['GET'])
def index():
    """Simple index page with API documentation"""
    return """
    <h1>Orpheus TTS Streaming Server</h1>
    <p>Original Orpheus TTS Model with Streaming Audio</p>
    
    <h2>Endpoints:</h2>
    <ul>
        <li><strong>GET /health</strong> - Health check</li>
        <li><strong>GET /tts?prompt=text</strong> - Generate TTS (streaming WAV)</li>
        <li><strong>POST /tts</strong> - Generate TTS with JSON payload</li>
        <li><strong>GET /tts_test</strong> - Quick test with default prompt</li>
    </ul>
    
    <h2>Example:</h2>
    <p><a href="/tts_test">Test TTS</a></p>
    <p><a href="/tts?prompt=Hello world, this is Orpheus TTS">Custom TTS</a></p>
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
    host = os.environ.get('TTS_HOST', '0.0.0.0')
    port = int(os.environ.get('TTS_PORT', 5005))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting Orpheus TTS Streaming Server on {host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    # Run Flask app
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Important: disable reloader with multiprocessing
    )

if __name__ == '__main__':
    main()
