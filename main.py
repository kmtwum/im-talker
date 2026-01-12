import os
import subprocess
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IMTalker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_video(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
    crop: bool = Form(True)
):
    """Generate talking face video from audio using default avatar"""
    
    # Use default avatar image
    img_path = "/app/img/avatar.jpg"
    aud_path = "/app/aud/audio.wav"
    
    # Generate unique output filename
    output_id = user_id
    output_dir = f"/app/results/{output_id}/"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Build command
        cmd = [
            "python3", "generator/generate.py",
            "--ref_path", img_path,
            "--aud_path", aud_path,
            "--res_dir", output_dir,
            "--generator_path", "/app/checkpoints/generator.ckpt",
            "--renderer_path", "/app/checkpoints/renderer.ckpt",
            "--a_cfg_scale", "3"
        ]
        
        if crop:
            cmd.append("--crop")
        
        # Run inference
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
        
        if result.returncode != 0:
            raise Exception(f"Generation failed: {result.stderr}")
        
        # Find generated video file
        video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
        if not video_files:
            raise Exception("No video file generated")
        
        video_path = os.path.join(output_dir, video_files[0])
        
        # Clean up temp file
        os.unlink(aud_path)
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"generated_{output_id}.mp4"
        )
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}