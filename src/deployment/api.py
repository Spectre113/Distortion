"""
FastAPI server for audio denoising inference.
"""
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple
import io

import torch
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Try to import moviepy for video support (optional)
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent
pipeline_path = project_root / "src" / "pipeline"
sys.path.insert(0, str(pipeline_path))

# Import from inference module
from inference import (
    load_best_model,
    split_audio_into_chunks,
    reconstruct_from_chunks
)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Denoising API",
    description="API for guitar audio denoising using WaveUNet",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global model and device
model = None
device = None
model_checkpoint_path = None
MODEL_VERSION = "v1"


def get_model():
    """Lazy loading of model - loads on first request."""
    global model, device, model_checkpoint_path
    
    if model is None:
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        checkpoint_path = project_root / "models" / "trained" / "waveunet_guitar_denoising_v1.pt"
        
        if not checkpoint_path.exists():
            raise RuntimeError(f"Model checkpoint not found at {checkpoint_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_best_model(str(checkpoint_path), device=device)
        model_checkpoint_path = str(checkpoint_path)
        print(f"Model loaded on device: {device}")
    
    return model, device


def process_audio_in_memory(
    audio_data: np.ndarray,
    sample_rate: int,
    model: torch.nn.Module,
    device: torch.device,
    chunk_duration: float = 3.0,
    batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process audio in memory without saving to disk.
    
    Returns:
        target_audio: denoised target audio
        residual_audio: extracted noise/residual
    """
    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Split into chunks
    chunks, orig_len = split_audio_into_chunks(audio_data, sample_rate, chunk_duration)
    
    preds_target = []
    preds_residual = []
    
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_arr = np.stack(batch_chunks, axis=0)  # (B, L)
            batch_tensor = torch.from_numpy(batch_arr).float().unsqueeze(1).to(device)  # (B,1,L)
            
            input_size = model.shapes["input_frames"]
            
            # Padding
            sample_diff = input_size - batch_tensor.shape[-1]
            if sample_diff > 0:
                batch_tensor = torch.nn.functional.pad(batch_tensor, (sample_diff // 2, 0))
                batch_tensor = torch.nn.functional.pad(batch_tensor, (0, sample_diff - sample_diff // 2))
            else:
                raise ValueError(f"Input size {batch_tensor.shape[-1]} is larger than model input size {input_size}")
            
            out_dict = model(batch_tensor)
            target_est, residual_est = out_dict["target"], out_dict["residual"]
            target_tensor = target_est.detach().cpu().numpy()  # (B,1,L)
            residual_tensor = residual_est.detach().cpu().numpy()
            
            out_arrs_target = [o[0].astype(np.float32) for o in target_tensor]
            out_arrs_residual = [o[0].astype(np.float32) for o in residual_tensor]
            preds_target.extend(out_arrs_target)
            preds_residual.extend(out_arrs_residual)
    
    # Reconstruct
    reconstructed_target = reconstruct_from_chunks(preds_target, orig_len)
    reconstructed_residual = reconstruct_from_chunks(preds_residual, orig_len)
    
    return reconstructed_target, reconstructed_residual


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        get_model()
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        model, device = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device) if device else None
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/version")
async def get_version():
    """Get API and model version."""
    return {
        "api_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "checkpoint": model_checkpoint_path if model_checkpoint_path else "not loaded"
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page."""
    static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.post("/upload")
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    return_type: str = "target"  # "target", "residual", or "both"
):
    """
    Upload audio or video file and process it.
    
    Parameters:
    - file: Audio file (WAV, MP3, FLAC, OGG, M4A) or Video file (MP4, AVI, MOV, MKV, WEBM, FLV)
    - return_type: "target" (denoised), "residual" (noise), or "both"
    
    For video files, audio will be extracted automatically.
    Returns processed audio file.
    """
    # Validate file type - support audio and video files
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
    
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    is_audio = file_ext in audio_extensions
    is_video = file_ext in video_extensions
    
    if not is_audio and not is_video:
        # Check content type as fallback
        if not file.content_type or not any(
            file.content_type.startswith(t) for t in ["audio/", "video/", "application/octet-stream"]
        ):
            raise HTTPException(
                status_code=400,
                detail=f"File must be an audio file ({', '.join(audio_extensions)}) or video file ({', '.join(video_extensions)})"
            )
    
    try:
        # Load model
        model, device = get_model()
        
        # Read uploaded file
        contents = await file.read()
        
        # Determine file extension
        file_ext = Path(file.filename).suffix.lower() if file.filename else '.wav'
        is_video = file_ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
        
        # Save to temporary file
        suffix = file_ext if file_ext else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract audio from video if needed
            if is_video:
                if not MOVIEPY_AVAILABLE:
                    raise HTTPException(
                        status_code=400,
                        detail="Video file support requires moviepy. Install with: pip install moviepy"
                    )
                
                # Extract audio from video
                video = VideoFileClip(tmp_path)
                audio_path = tmp_path.replace(suffix, '_audio.wav')
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                video.close()
                
                # Load extracted audio
                audio, sr = librosa.load(audio_path, sr=22050, mono=True)
                
                # Clean up extracted audio file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            else:
                # Load audio directly
                audio, sr = librosa.load(tmp_path, sr=22050, mono=True)
            
            # Process audio
            target_audio, residual_audio = process_audio_in_memory(
                audio_data=audio,
                sample_rate=sr,
                model=model,
                device=device,
                chunk_duration=3.0,
                batch_size=8
            )
            
            # Determine which audio to return
            if return_type == "target":
                output_audio = target_audio
                output_filename = f"denoised_{file.filename}"
            elif return_type == "residual":
                output_audio = residual_audio
                output_filename = f"residual_{file.filename}"
            elif return_type == "both":
                # Return both as a zip or return target by default
                output_audio = target_audio
                output_filename = f"denoised_{file.filename}"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="return_type must be 'target', 'residual', or 'both'"
                )
            
            # Save processed audio to temporary file
            # Use a unique name to avoid conflicts
            output_path = tmp_path.rsplit('.', 1)[0] + "_processed.wav"
            sf.write(output_path, output_audio, sr)
            
            # Clean up input temp file before sending response
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            # Schedule cleanup of output file after response is sent
            def cleanup_output_file():
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass
            
            background_tasks.add_task(cleanup_output_file)
            
            return FileResponse(
                output_path,
                media_type="audio/wav",
                filename=output_filename
            )
            
        except Exception as e:
            # Clean up on error
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            if 'output_path' in locals() and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            raise
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/process")
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    return_type: str = "target"
):
    """
    Alias for /upload endpoint.
    """
    return await upload_and_process(background_tasks, file, return_type)


if __name__ == "__main__":
    uvicorn.run(
        "src.deployment.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

