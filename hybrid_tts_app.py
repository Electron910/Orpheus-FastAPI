#!/usr/bin/env python3
"""
Hybrid TTS Application - Integrates Original Orpheus TTS with existing FastAPI structure
This replaces the problematic token-based approach with direct Orpheus TTS
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import sys
import struct
import asyncio
import multiprocessing
from typing import Optional, Generator
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="Orpheus TTS Hybrid API", version="2.0.0")

# Mount static files and templates
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")

# Global model instance
orpheus_engine = None

class TTSRequest(BaseModel):
    prompt: str
    voice: Optional[str] = "tara"
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2000
    repetition_penalty: Optional[float] = 1.1

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

async def initialize_orpheus_model():
    """Initialize the Orpheus TTS model asynchronously"""
    global orpheus_engine
    
    if orpheus_engine is not None:
        return True
        
    try:
        print("🎤 Initializing Original Orpheus TTS Model...")
        from orpheus_tts import OrpheusModel
        
        model_name = os.environ.get('ORPHEUS_MODEL_NAME', 'canopylabs/orpheus-tts-0.1-finetune-prod')
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        orpheus_engine = await loop.run_in_executor(
            None, 
            lambda: OrpheusModel(model_name=model_name)
        )
        
        print("✅ Orpheus TTS Model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Orpheus TTS model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    await initialize_orpheus_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Orpheus TTS Hybrid API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007acc; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🎤 Orpheus TTS Hybrid API</h1>
        <p>Original Orpheus TTS Model with FastAPI Integration</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/health</strong><br>
            Health check and model status
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <strong>/generate_speech</strong><br>
            Generate TTS audio (streaming WAV)<br>
            Body: {"prompt": "text", "voice": "tara", "temperature": 0.4}
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/tts</strong><br>
            Quick TTS generation with query parameters<br>
            Example: /tts?prompt=Hello%20world&voice=tara
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/test</strong><br>
            Test endpoint with sample audio
        </div>
        
        <h2>Quick Test:</h2>
        <p><a href="/test" target="_blank">🎵 Generate Test Audio</a></p>
        <p><a href="/tts?prompt=Hello%20world,%20this%20is%20the%20new%20Orpheus%20TTS%20system" target="_blank">🗣️ Custom TTS</a></p>
        
        <h2>Model Status:</h2>
        <p><a href="/health">Check Model Health</a></p>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if orpheus_engine is None:
        return {"status": "error", "message": "Orpheus TTS model not initialized"}
    return {
        "status": "ok", 
        "model": "orpheus-tts",
        "engine": "original-orpheus",
        "version": "2.0.0"
    }

async def generate_audio_stream(prompt: str, voice: str = "tara", temperature: float = 0.4, 
                               top_p: float = 0.9, max_tokens: int = 2000, 
                               repetition_penalty: float = 1.1) -> Generator[bytes, None, None]:
    """Generate streaming audio from text"""
    if orpheus_engine is None:
        raise HTTPException(status_code=500, detail="Orpheus TTS model not initialized")
    
    try:
        print(f"🗣️ Generating TTS for: '{prompt[:50]}...' with voice: {voice}")
        
        # Send WAV header first
        yield create_wav_header()
        
        # Generate speech tokens using original Orpheus
        loop = asyncio.get_event_loop()
        syn_tokens = await loop.run_in_executor(
            None,
            lambda: orpheus_engine.generate_speech(
                prompt=prompt,
                voice=voice,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
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
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/generate_speech")
async def generate_speech_endpoint(request: TTSRequest):
    """Main TTS generation endpoint (POST)"""
    return StreamingResponse(
        generate_audio_stream(
            prompt=request.prompt,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty
        ),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )

@app.get("/tts")
async def tts_get_endpoint(
    prompt: str,
    voice: str = "tara",
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 2000,
    repetition_penalty: float = 1.1
):
    """TTS generation endpoint (GET) for easy testing"""
    return StreamingResponse(
        generate_audio_stream(
            prompt=prompt,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        ),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )

@app.get("/test")
async def test_endpoint():
    """Test endpoint with sample audio"""
    test_prompt = "Hello! This is a test of the new Orpheus TTS hybrid system. The original model is now working correctly with proper audio generation."
    
    return StreamingResponse(
        generate_audio_stream(prompt=test_prompt, voice="tara"),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=test_speech.wav"}
    )

# Legacy compatibility endpoints
@app.post("/api/tts")
async def legacy_tts_endpoint(request: TTSRequest):
    """Legacy TTS endpoint for backward compatibility"""
    return await generate_speech_endpoint(request)

def main():
    """Main function with proper multiprocessing setup"""
    # Set multiprocessing method for VLLM compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    # Get configuration from environment
    host = os.environ.get('TTS_HOST', '0.0.0.0')
    port = int(os.environ.get('TTS_PORT', 5005))
    
    print(f"🚀 Starting Orpheus TTS Hybrid Server on {host}:{port}")
    print("🔧 Using Original Orpheus TTS Model")
    
    # Run with uvicorn
    uvicorn.run(
        "hybrid_tts_app:app",
        host=host,
        port=port,
        reload=False,  # Important: disable reload with multiprocessing
        workers=1
    )

if __name__ == '__main__':
    main()
