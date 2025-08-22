import os
import time
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from tts_engine import (
    generate_speech_from_api,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    VOICE_TO_LANGUAGE,
    AVAILABLE_LANGUAGES,
    generate_audio_stream,
    generate_tokens_from_api
)

# Load .env environment variables
load_dotenv(override=True)

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(
    title="Orpheus-FASTAPI",
    description="High-performance Text-to-Speech server using Orpheus-FASTAPI",
    version="1.3.1"
)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0

### MAIN STREAMING ENDPOINT ###
@app.post("/v1/audio/speech")
async def create_speech_api_streaming(request: SpeechRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    async def async_token_gen():
        sync_gen = generate_tokens_from_api(
            prompt=request.input,
            voice=request.voice,
            temperature=float(os.environ.get("ORPHEUS_TEMPERATURE", "0.6")),
            top_p=float(os.environ.get("ORPHEUS_TOP_P", "0.9")),
            max_tokens=int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192")),
            repetition_penalty=1.1
        )
        for token in sync_gen:
            yield token
    async def stream_audio_response():
        async for audio_chunk in generate_audio_stream(async_token_gen()):
            yield audio_chunk
    return StreamingResponse(
        stream_audio_response(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{request.voice}_{int(time.time())}.wav"',
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked"
        }
    )

# FILE-BASED, NON-STREAMING (for backward compatibility)
@app.post("/v1/audio/speech/file")
async def create_speech_api_file(request: SpeechRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    use_batching = len(request.input) > 1000
    generate_speech_from_api(
        prompt=request.input,
        voice=request.voice,
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"{request.voice}_{timestamp}.wav"
    )

@app.get("/v1/audio/voices")
async def list_voices():
    return JSONResponse(
        content={
            "status": "ok",
            "voices": AVAILABLE_VOICES
        }
    )

# STREAMING LEGACY API ENDPOINT
@app.post("/speak")
async def speak_streaming(request: Request):
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)
    if not text:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'text'"}
        )
    async def async_token_gen():
        sync_gen = generate_tokens_from_api(
            prompt=text,
            voice=voice,
            temperature=float(os.environ.get("ORPHEUS_TEMPERATURE", "0.6")),
            top_p=float(os.environ.get("ORPHEUS_TOP_P", "0.9")),
            max_tokens=int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192")),
            repetition_penalty=1.1
        )
        for token in sync_gen:
            yield token
    async def stream_audio_response():
        async for audio_chunk in generate_audio_stream(async_token_gen()):
            yield audio_chunk
    return StreamingResponse(
        stream_audio_response(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{voice}_{int(time.time())}.wav"',
            "Cache-Control": "no-cache"
        }
    )

# NON-STREAMING LEGACY API ENDPOINT (for file saving)
@app.post("/speak/file")
async def speak_file(request: Request):
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)
    if not text:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'text'"}
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    use_batching = len(text) > 1000
    generate_speech_from_api(
        prompt=text,
        voice=voice,
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    return JSONResponse(content={
        "status": "ok",
        "voice": voice,
        "output_file": output_path,
        "generation_time": 0
    })

# --- Web UI routes ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "voices": AVAILABLE_VOICES,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

@app.get("/web/", response_class=HTMLResponse)
async def web_ui(request: Request):
    config = get_current_config()
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "voices": AVAILABLE_VOICES,
            "config": config,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

@app.get("/get_config")
async def get_config():
    config = get_current_config()
    return JSONResponse(content=config)

@app.post("/save_config")
async def save_config(request: Request):
    data = await request.json()
    # Convert values to strings for saving to .env
    for key, value in data.items():
        if key in ["ORPHEUS_MAX_TOKENS", "ORPHEUS_API_TIMEOUT", "ORPHEUS_PORT", "ORPHEUS_SAMPLE_RATE"]:
            try:
                data[key] = str(int(value))
            except (ValueError, TypeError):
                pass
        elif key in ["ORPHEUS_TEMPERATURE", "ORPHEUS_TOP_P"]:
            try:
                data[key] = str(float(value))
            except (ValueError, TypeError):
                pass
    # Write configuration to .env file
    with open(".env", "w") as f:
        for key, value in data.items():
            f.write(f"{key}={value}\n")
    return JSONResponse(content={"status": "ok", "message": "Configuration saved successfully. Restart server to apply changes."})

@app.post("/restart_server")
async def restart_server():
    import threading
    def touch_restart_file():
        time.sleep(0.5)
        restart_file = "restart.flag"
        with open(restart_file, "w") as f:
            f.write(str(time.time()))
        print("🔄 Restart flag created, server will reload momentarily...")
    threading.Thread(target=touch_restart_file, daemon=True).start()
    return JSONResponse(content={"status": "ok", "message": "Server is restarting. Please wait a moment..."})

def get_current_config():
    default_config = {}
    if os.path.exists(".env.example"):
        with open(".env.example", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    default_config[key] = value
    current_config = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    current_config[key] = value
    config = {**default_config, **current_config}
    for key in config:
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value
    return config

@app.post("/web/", response_class=HTMLResponse)
async def generate_from_web(
    request: Request,
    text: str = Form(...),
    voice: str = Form(DEFAULT_VOICE)
):
    if not text:
        return templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "error": "Please enter some text.",
                "voices": AVAILABLE_VOICES,
                "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
                "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
            }
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    use_batching = len(text) > 1000
    generate_speech_from_api(
        prompt=text,
        voice=voice,
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "success": True,
            "text": text,
            "voice": voice,
            "output_file": output_path,
            "generation_time": 0,
            "voices": AVAILABLE_VOICES,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("ORPHEUS_HOST", "0.0.0.0")
    port = int(os.environ.get("ORPHEUS_PORT", "5005"))
    print(f"🔥 Starting Orpheus-FASTAPI Server on {host}:{port}")
    print(f"💬 Web UI available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"📖 API docs available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    print(f"🚀 STREAMING TTS available at /v1/audio/speech (real-time)")
    print(f"💾 FILE TTS available at /v1/audio/speech/file (saves to disk)")
    uvicorn.run("app:app", host=host, port=port, reload=True)
