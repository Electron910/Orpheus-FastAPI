#!/usr/bin/env python3
"""
PRODUCTION Orpheus TTS Server - VLLM EngineDeadError FIX
Implements proven workarounds for VLLM engine stability issues
"""

import os
import sys
import time
import uuid
import threading
import multiprocessing
from datetime import datetime
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import logging

# Set multiprocessing method BEFORE importing orpheus_tts
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Import after setting spawn method
from orpheus_tts import OrpheusModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ProductionOrpheusEngine:
    def __init__(self):
        self.model = None
        self.model_lock = threading.Lock()
        self.request_count = 0
        self.max_requests_before_restart = 10  # Restart engine every 10 requests
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Orpheus TTS model with VLLM stability fixes"""
        try:
            logger.info("🎤 Initializing Production Orpheus TTS Model...")
            
            # Get model name from environment
            model_name = os.getenv('ORPHEUS_MODEL_NAME', 'canopylabs/orpheus-tts-0.1-finetune-prod')
            
            # CRITICAL: Apply VLLM stability fixes
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            os.environ['VLLM_DISABLE_CUSTOM_ALL_REDUCE'] = '1'  # KEY FIX for EngineDeadError
            os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '300'  # Increase timeout
            
            # Initialize with conservative settings
            self.model = OrpheusModel(
                model_name=model_name,
                # Add any additional stability parameters here
            )
            
            logger.info("✅ Production Orpheus TTS Model initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Orpheus TTS model: {e}")
            self.model = None
            return False
    
    def restart_model_if_needed(self):
        """Restart model periodically to prevent engine death"""
        self.request_count += 1
        
        if self.request_count >= self.max_requests_before_restart:
            logger.info(f"🔄 Restarting model after {self.request_count} requests (preventive maintenance)")
            
            # Clean shutdown
            if self.model:
                try:
                    del self.model
                except:
                    pass
            
            # Reinitialize
            time.sleep(2)  # Brief pause
            self.initialize_model()
            self.request_count = 0
    
    def generate_speech(self, text, voice="tara", temperature=0.4, top_p=0.9, 
                       max_tokens=1500, repetition_penalty=1.1):
        """Generate speech with engine stability management"""
        
        if not self.model:
            raise Exception("TTS model not initialized")
        
        with self.model_lock:
            try:
                # Check if we need preventive restart
                self.restart_model_if_needed()
                
                # Generate unique request ID
                request_id = f"req-{uuid.uuid4().hex[:8]}"
                
                logger.info(f"🗣️ Generating TTS for: '{text[:50]}...' with voice: {voice}")
                logger.info(f"📝 Request ID: {request_id}")
                
                # Generate speech with timeout protection
                audio_chunks = []
                chunk_count = 0
                
                start_time = time.time()
                timeout = 60  # 60 second timeout per request
                
                for chunk in self.model.generate(
                    prompt=text,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    request_id=request_id
                ):
                    # Check timeout
                    if time.time() - start_time > timeout:
                        logger.error(f"⏰ Request {request_id} timed out after {timeout}s")
                        raise Exception(f"Request timed out after {timeout} seconds")
                    
                    audio_chunks.append(chunk)
                    chunk_count += 1
                    
                    if chunk_count % 10 == 0:
                        logger.info(f"📦 Processed {chunk_count} chunks...")
                
                # Combine all chunks
                full_audio = b''.join(audio_chunks)
                
                generation_time = time.time() - start_time
                logger.info(f"✅ Generated {chunk_count} audio chunks, total size: {len(full_audio)} bytes")
                logger.info(f"⏱️ Generation time: {generation_time:.2f}s")
                
                return full_audio
                
            except Exception as e:
                logger.error(f"❌ TTS generation failed: {e}")
                
                # If engine died, try to reinitialize
                if "engine dead" in str(e).lower() or "enginedeaderror" in str(e).lower():
                    logger.warning("🔄 Engine died, attempting recovery...")
                    self.model = None
                    time.sleep(1)
                    if self.initialize_model():
                        logger.info("✅ Engine recovered, retrying request...")
                        # Retry once after recovery
                        return self.generate_speech(text, voice, temperature, top_p, max_tokens, repetition_penalty)
                
                raise

# Global TTS engine
tts_engine = None

def initialize_tts():
    """Initialize TTS engine"""
    global tts_engine
    if not tts_engine:
        tts_engine = ProductionOrpheusEngine()
    return tts_engine.model is not None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if tts_engine and tts_engine.model else "unhealthy",
        "model_loaded": tts_engine and tts_engine.model is not None,
        "request_count": tts_engine.request_count if tts_engine else 0,
        "max_requests_before_restart": tts_engine.max_requests_before_restart if tts_engine else 0,
        "timestamp": datetime.now().isoformat(),
        "version": "production-v1.0",
        "vllm_fixes_applied": True
    })

@app.route('/tts', methods=['GET'])
def generate_tts():
    """Generate TTS audio with production stability"""
    try:
        # Get parameters
        text = request.args.get('prompt', '').strip()
        voice = request.args.get('voice', 'tara')
        temperature = float(request.args.get('temperature', 0.4))
        top_p = float(request.args.get('top_p', 0.9))
        max_tokens = int(request.args.get('max_tokens', 1500))
        repetition_penalty = float(request.args.get('repetition_penalty', 1.1))
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if not tts_engine or not tts_engine.model:
            return jsonify({"error": "TTS model not available"}), 503
        
        # Generate speech
        audio_data = tts_engine.generate_speech(
            text=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )
        
        # Return audio
        return Response(
            audio_data,
            mimetype='audio/wav',
            headers={
                'Content-Disposition': 'attachment; filename="speech.wav"',
                'Content-Length': str(len(audio_data)),
                'X-Request-Count': str(tts_engine.request_count),
                'X-Engine-Status': 'healthy'
            }
        )
        
    except Exception as e:
        logger.error(f"❌ TTS request failed: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Try again in a few seconds"
        }), 500

@app.route('/test', methods=['GET'])
def test_tts():
    """Test TTS with sample text"""
    try:
        test_text = "Hello! This is a production test of the Orpheus TTS system with VLLM stability fixes."
        audio_data = tts_engine.generate_speech(test_text)
        
        return Response(
            audio_data,
            mimetype='audio/wav',
            headers={
                'Content-Disposition': 'attachment; filename="test.wav"'
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/restart', methods=['POST'])
def restart_engine():
    """Manually restart the TTS engine"""
    try:
        if tts_engine:
            logger.info("🔄 Manual engine restart requested")
            tts_engine.model = None
            time.sleep(2)
            success = tts_engine.initialize_model()
            tts_engine.request_count = 0
            
            return jsonify({
                "status": "restarted" if success else "failed",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "No engine to restart"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Production web interface"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Production Orpheus TTS Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            input, button { padding: 12px; margin: 5px; border-radius: 5px; border: 1px solid #ddd; }
            #textInput { width: 500px; }
            button { background: #007bff; color: white; border: none; cursor: pointer; font-weight: bold; }
            button:hover { background: #0056b3; }
            .danger { background: #dc3545; }
            .danger:hover { background: #c82333; }
            audio { width: 100%; margin: 20px 0; }
            .info { background: #d1ecf1; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 Production Orpheus TTS Server</h1>
                <p>VLLM EngineDeadError Fixed • Production Ready</p>
            </div>
            
            <div class="info">
                <strong>🛠️ Applied Fixes:</strong><br>
                • VLLM_DISABLE_CUSTOM_ALL_REDUCE=1<br>
                • Spawn multiprocessing method<br>
                • Preventive engine restarts<br>
                • Engine death recovery<br>
                • Request timeout protection
            </div>
            
            <div>
                <input type="text" id="textInput" placeholder="Enter text to speak..." 
                       value="Hello! This is a production test of the Orpheus TTS system.">
                <button onclick="generateSpeech()">🗣️ Generate Speech</button>
            </div>
            
            <div>
                <audio id="audioPlayer" controls></audio>
            </div>
            
            <div style="margin-top: 20px;">
                <button onclick="testTTS()">🧪 Test TTS</button>
                <button onclick="checkHealth()">❤️ Health Check</button>
                <button onclick="restartEngine()" class="danger">🔄 Restart Engine</button>
            </div>
            
            <div id="status" class="status"></div>
        </div>

        <script>
            function generateSpeech() {
                const text = document.getElementById('textInput').value;
                if (!text.trim()) {
                    alert('Please enter some text');
                    return;
                }
                
                const url = '/tts?prompt=' + encodeURIComponent(text);
                document.getElementById('audioPlayer').src = url;
                document.getElementById('status').innerHTML = '🗣️ Generating speech with production stability...';
            }
            
            function testTTS() {
                document.getElementById('audioPlayer').src = '/test';
                document.getElementById('status').innerHTML = '🧪 Testing production TTS...';
            }
            
            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('status').innerHTML = 
                        `<strong>🏥 Health Status:</strong><br>
                         Status: ${data.status}<br>
                         Model Loaded: ${data.model_loaded}<br>
                         Request Count: ${data.request_count}/${data.max_requests_before_restart}<br>
                         VLLM Fixes Applied: ${data.vllm_fixes_applied}<br>
                         Version: ${data.version}<br>
                         Time: ${data.timestamp}`;
                } catch (error) {
                    document.getElementById('status').innerHTML = 
                        `<strong>❌ Error:</strong> ${error.message}`;
                }
            }
            
            async function restartEngine() {
                if (!confirm('Are you sure you want to restart the TTS engine?')) return;
                
                try {
                    document.getElementById('status').innerHTML = '🔄 Restarting engine...';
                    const response = await fetch('/restart', { method: 'POST' });
                    const data = await response.json();
                    document.getElementById('status').innerHTML = 
                        `<strong>🔄 Engine Restart:</strong><br>
                         Status: ${data.status}<br>
                         Time: ${data.timestamp}`;
                } catch (error) {
                    document.getElementById('status').innerHTML = 
                        `<strong>❌ Restart Error:</strong> ${error.message}`;
                }
            }
            
            // Auto-check health on page load
            window.onload = function() {
                checkHealth();
            };
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Starting PRODUCTION Orpheus TTS Server...")
    print("🛠️ VLLM EngineDeadError fixes applied:")
    print("   • VLLM_DISABLE_CUSTOM_ALL_REDUCE=1")
    print("   • Spawn multiprocessing method")
    print("   • Preventive engine restarts")
    print("   • Engine death recovery")
    print("   • Request timeout protection")
    
    # Initialize TTS
    if initialize_tts():
        print("✅ TTS initialized successfully")
        print("🌐 Server starting on http://0.0.0.0:5005")
        print("🔗 Your URL: https://561pjq4x4ud1px-5005.proxy.runpod.net/")
        app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)
    else:
        print("❌ Failed to initialize TTS. Exiting.")
        sys.exit(1)
