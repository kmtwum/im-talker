import os
import tempfile
import subprocess
import glob
from typing import Optional, List, AsyncGenerator
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import torch
from torch.amp import autocast
import numpy as np
import cv2
import librosa
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from transformers import Wav2Vec2FeatureExtractor
import face_alignment

# ==== Latency Optimizations ====
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(True)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(True)

from generator.FM import FMGenerator
from renderer.models import IMTRenderer

app = FastAPI(title="IMTalker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceConfig:
    """Configuration matching base_options.py defaults"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rank = self.device
        self.seed = 42
        self.fix_noise_seed = False
        self.input_size = 512
        self.input_nc = 3
        self.fps = 25.0
        self.sampling_rate = 16000
        self.audio_marcing = 2
        self.wav2vec_sec = 2.0
        self.wav2vec_model_path = "/app/checkpoints/wav2vec2-base-960h"
        self.attention_window = 5
        self.only_last_features = True
        self.audio_dropout_prob = 0.1
        self.style_dim = 512
        self.dim_a = 512
        self.dim_h = 512
        self.dim_e = 7
        self.dim_motion = 32
        self.dim_c = 32
        self.dim_w = 32
        self.fmt_depth = 8
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.no_learned_pe = False
        self.num_prev_frames = 5  # Reduced from 10 for faster processing
        # Optimized defaults
        self.ode_atol = 1e-4
        self.ode_rtol = 1e-4
        self.nfe = 7
        self.torchdiffeq_ode_method = 'euler'
        # CFG scale: 1.0 = no CFG (fastest), >1.0 = 2x ODE computation per step
        self.a_cfg_scale = 1.5  # Reduced from 3.0 - higher values double computation
        self.swin_res_threshold = 128
        self.window_size = 8
        # Paths
        self.renderer_path = "/app/checkpoints/renderer.ckpt"
        self.generator_path = "/app/checkpoints/generator.ckpt"


class InferenceAgent:
    """Persistent inference agent - models loaded once at startup"""
    
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.device = opt.device
        
        print("Loading models (one-time at startup)...")
        self.renderer = IMTRenderer(opt).to(self.device)
        self.generator = FMGenerator(opt).to(self.device)
        
        self._load_renderer(opt.renderer_path)
        self._load_generator(opt.generator_path)
        
        self.renderer.eval()
        self.generator.eval()
        
        # Pre-load face alignment and wav2vec (one-time)
        print("Loading face alignment...")
        fa_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            device=fa_device, 
            flip_input=False
        )
        
        print("Loading Wav2Vec2 preprocessor...")
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            opt.wav2vec_model_path, local_files_only=True
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        print(f"Inference agent ready! {opt.device}")
    
    def _load_renderer(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_dict = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
        self.renderer.load_state_dict(clean_dict, strict=False)
    
    def _load_generator(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        prefix = 'model.'
        clean_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        with torch.no_grad():
            for name, param in self.generator.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name])  # Already on correct device
    
    def process_image(self, img_path: str, crop: bool = True) -> torch.Tensor:
        """Load and preprocess source image"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        
        if crop:
            img_arr = np.array(img_pil)
            bboxes = self.fa.face_detector.detect_from_image(img_arr)
            valid_bboxes = [
                (int(x1), int(y1), int(x2), int(y2), score)
                for (x1, y1, x2, y2, score) in bboxes if score > 0.95
            ]
            if valid_bboxes:
                x1, y1, x2, y2, _ = valid_bboxes[0]
                h, w = img_arr.shape[:2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half = int(max(x2 - x1, y2 - y1) * 0.8)
                x1_new = max(cx - half, 0)
                x2_new = min(cx + half, w)
                y1_new = max(cy - half, 0)
                y2_new = min(cy + half, h)
                side = min(x2_new - x1_new, y2_new - y1_new)
                x2_new = x1_new + side
                y2_new = y1_new + side
                crop_img = img_arr[y1_new:y2_new, x1_new:x2_new]
                img_pil = Image.fromarray(crop_img)
        
        return self.transform(img_pil).unsqueeze(0).to(self.device, non_blocking=True)
    
    def process_audio(self, aud_path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        speech_array, sr = librosa.load(aud_path, sr=self.opt.sampling_rate)
        return self.wav2vec_preprocessor(
            speech_array, sampling_rate=sr, return_tensors='pt'
        ).input_values[0].unsqueeze(0).to(self.device, non_blocking=True)
    
    @torch.no_grad()
    def generate(self, img_path: str, aud_path: str, output_path: str, 
                 crop: bool = True, cfg_scale: float = 3.0, nfe: int = 7, output_size: int = 512) -> str:
        """Run inference and return video path
        
        Args:
            output_size: Output video size (width=height, 1:1 ratio). Default 512.
        """
        
        # Preprocess inputs
        s_tensor = self.process_image(img_path, crop)
        a_tensor = self.process_audio(aud_path)
        
        # Encode source image (done once, reused for all frames)
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        
        # Prepare data for generator
        data = {
            's': s_tensor,
            'a': a_tensor,
            'pose': None,
            'cam': None,
            'gaze': None,
            'ref_x': t_lat
        }
        
        # Generate motion latents
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=self.opt.seed)
        
        # Decode to frames - simple loop (batching doesn't help here due to memory constraints)
        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        d_hat = []
        with autocast(dtype=torch.bfloat16):
            for t in range(T):
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                d_hat.append(self.renderer.decode(m_c, m_r, f_r))
        
        vid_tensor = torch.stack(d_hat, dim=1).squeeze()
        
        # Ensure CUDA operations complete before video save
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Save video (resize on GPU if needed)
        return self._save_video(vid_tensor, output_path, aud_path, output_size)
    
    def _save_video(self, vid_tensor, output_path, audio_path, output_size: int = 512):
        """Save video with audio, resizing on GPU if needed.
        
        Args:
            output_size: Target size for output video (1:1 ratio)
        """
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            temp_path = tmp.name
        
        vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
        vid = (vid * 255).type(torch.ByteTensor)
        torchvision.io.write_video(temp_path, vid, fps=self.opt.fps)
        
        if audio_path:
            cmd = f'ffmpeg -i {temp_path} -i {audio_path} -c:v copy -c:a aac {output_path} -y -loglevel error'
            subprocess.call(cmd, shell=True)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            import shutil
            shutil.move(temp_path, output_path)
        
        return output_path
    
    @torch.no_grad()
    def generate_chunk(self, img_tensor: torch.Tensor, aud_path: str, 
                       f_r: torch.Tensor, g_r: torch.Tensor, t_lat: torch.Tensor,
                       cfg_scale: float = 3.0, nfe: int = 7) -> bytes:
        """Generate video for a single audio chunk and return MP4 bytes.
        
        Args:
            img_tensor: Pre-processed source image tensor
            aud_path: Path to the audio chunk
            f_r, g_r, t_lat: Pre-computed renderer encodings (reused across chunks)
            cfg_scale: CFG scale for generation
            nfe: Number of function evaluations
            
        Returns:
            MP4 video bytes with muxed audio
        """
        # Process audio chunk
        a_tensor = self.process_audio(aud_path)
        
        # Prepare data for generator
        data = {
            's': img_tensor,
            'a': a_tensor,
            'pose': None,
            'cam': None,
            'gaze': None,
            'ref_x': t_lat
        }
        
        # Generate motion latents for this chunk
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=self.opt.seed)
        
        # Decode to frames
        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        d_hat = []
        with autocast(dtype=torch.bfloat16):
            for t in range(T):
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                d_hat.append(self.renderer.decode(m_c, m_r, f_r))
        
        vid_tensor = torch.stack(d_hat, dim=1).squeeze()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Convert to video bytes
        return self._encode_to_mp4_bytes(vid_tensor, aud_path)
    
    def _encode_to_mp4_bytes(self, vid_tensor: torch.Tensor, audio_path: str) -> bytes:
        """Encode video tensor to MP4 bytes with audio.
        
        Returns:
            MP4 video bytes
        """
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_vid:
            temp_vid_path = tmp_vid.name
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            temp_out_path = tmp_out.name
        
        try:
            # Write video frames
            vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
            vid = (vid * 255).type(torch.ByteTensor)
            torchvision.io.write_video(temp_vid_path, vid, fps=self.opt.fps)
            
            # Mux with audio
            cmd = f'ffmpeg -i {temp_vid_path} -i {audio_path} -c:v copy -c:a aac {temp_out_path} -y -loglevel error'
            subprocess.call(cmd, shell=True)
            
            # Read final MP4 bytes
            with open(temp_out_path, 'rb') as f:
                return f.read()
        finally:
            # Cleanup temp files
            if os.path.exists(temp_vid_path):
                os.remove(temp_vid_path)
            if os.path.exists(temp_out_path):
                os.remove(temp_out_path)


def split_audio_into_chunks(audio_path: str, chunk_duration: float, output_dir: str) -> List[str]:
    """Split audio file into chunks of specified duration using ffmpeg.
    
    Args:
        audio_path: Path to the source audio file
        chunk_duration: Duration of each chunk in seconds
        output_dir: Directory to save audio chunks
        
    Returns:
        List of paths to the audio chunks
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_pattern = os.path.join(output_dir, "chunk_%03d.wav")
    
    cmd = [
        'ffmpeg', '-i', audio_path,
        '-f', 'segment',
        '-segment_time', str(chunk_duration),
        '-c', 'copy',
        chunk_pattern,
        '-y', '-loglevel', 'error'
    ]
    subprocess.call(cmd)
    
    # Get list of generated chunks in order
    chunks = sorted(glob.glob(os.path.join(output_dir, "chunk_*.wav")))
    return chunks


def cleanup_chunks(chunk_paths: List[str], chunk_dir: str):
    """Clean up temporary audio chunks."""
    for path in chunk_paths:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(chunk_dir):
        try:
            os.rmdir(chunk_dir)
        except OSError:
            pass  # Directory not empty, leave it


# Initialize agent once at startup (models loaded here)
print("Initializing IMTalker inference agent...")
config = InferenceConfig()
agent = InferenceAgent(config)


@app.post("/generate")
async def generate_video(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    user_id: str = Form(...),
    crop: bool = Form(True),
    cfg_scale: float = Form(3.0),
    nfe: int = Form(7),
    size: int = Form(512, ge=64, le=512, description="Output video size in pixels (1:1 ratio)"),
    reference_aud_url: Optional[str] = Form(None),
    clone: Optional[str] = Form(None),
    split_sentences: bool = Form(False),
    speed: float = Form(1.0)
):
    """Generate talking face video from audio or text using default avatar.
    
    Provide either 'audio' (uploaded file) or 'text' (for TTS synthesis).
    When 'text' is provided, the TTS service at tts:8000/generate is called.
    
    This endpoint uses a persistent model - no subprocess overhead.
    """
    
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either 'audio' or 'text' must be provided")
    
    img_path = "/app/img/avatar_chest.jpg"
    output_dir = f"/app/results/{user_id}/"
    os.makedirs(output_dir, exist_ok=True)
    
    aud_path = f"/app/aud/{user_id}_audio.wav"
    os.makedirs(os.path.dirname(aud_path), exist_ok=True)
    
    try:
        if text:
            # Use TTS service to synthesize audio from text
            tts_payload = {
                "text": text,
                "source_aud": reference_aud_url or "",
                "split_sentences": split_sentences,
                "streaming": False,
                "speed": speed
            }
            if clone:
                tts_payload["clone"] = clone
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                tts_response = await client.post(
                    "http://tts:8000/generate",
                    json=tts_payload
                )
                if tts_response.status_code != 200:
                    raise HTTPException(
                        status_code=502, 
                        detail=f"TTS service error: {tts_response.text}"
                    )
                # Save the synthesized audio
                with open(aud_path, "wb") as f:
                    f.write(tts_response.content)
        else:
            # Use uploaded audio file
            with open(aud_path, "wb") as f:
                content = await audio.read()
                f.write(content)
            
        print("Audio saved to", aud_path)
        
        output_path = os.path.join(output_dir, f"{user_id}.mp4")
        
        # Direct inference - no subprocess!
        agent.generate(
            img_path=img_path,
            aud_path=aud_path,
            output_path=output_path,
            crop=crop,
            cfg_scale=cfg_scale,
            nfe=nfe,
            output_size=size
        )
        
        # Clean up temp audio
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"generated_{user_id}.mp4"
        )
        
    except httpx.RequestError as e:
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        raise HTTPException(status_code=502, detail=f"Failed to connect to TTS service: {str(e)}")
    except Exception as e:
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": agent is not None}


@app.post("/generate/stream")
async def generate_video_stream(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    user_id: str = Form(...),
    crop: bool = Form(True),
    cfg_scale: float = Form(3.0),
    nfe: int = Form(7),
    chunk_duration: float = Form(5.0, ge=2.0, le=15.0, description="Duration of each video chunk in seconds"),
    reference_aud_url: Optional[str] = Form(None),
    clone: Optional[str] = Form(None),
    split_sentences: bool = Form(False),
    speed: float = Form(1.0)
):
    """Stream video chunks as they're generated.
    
    Each chunk is a complete MP4 segment with synchronized audio.
    Chunks are yielded progressively as they complete, allowing the client
    to start playback before the full video is ready.
    
    Provide either 'audio' (uploaded file) or 'text' (for TTS synthesis).
    
    The response is a multipart stream where each part is a complete MP4 chunk.
    Chunks are separated by a boundary marker for easy parsing.
    """
    
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either 'audio' or 'text' must be provided")
    
    img_path = "/app/img/avatar_chest.jpg"
    aud_path = f"/app/aud/{user_id}_stream_audio.wav"
    chunk_dir = f"/app/aud/{user_id}_chunks/"
    os.makedirs(os.path.dirname(aud_path), exist_ok=True)
    
    # Get full audio first (from upload or TTS)
    try:
        if text:
            tts_payload = {
                "text": text,
                "source_aud": reference_aud_url or "",
                "split_sentences": split_sentences,
                "streaming": False,
                "speed": speed
            }
            if clone:
                tts_payload["clone"] = clone
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                tts_response = await client.post(
                    "http://tts:8000/generate",
                    json=tts_payload
                )
                if tts_response.status_code != 200:
                    raise HTTPException(
                        status_code=502, 
                        detail=f"TTS service error: {tts_response.text}"
                    )
                with open(aud_path, "wb") as f:
                    f.write(tts_response.content)
        else:
            with open(aud_path, "wb") as f:
                content = await audio.read()
                f.write(content)
    except httpx.RequestError as e:
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        raise HTTPException(status_code=502, detail=f"Failed to connect to TTS service: {str(e)}")
    
    async def chunk_generator() -> AsyncGenerator[bytes, None]:
        """Generate and yield video chunks."""
        chunk_paths = []
        try:
            # Pre-process image once (expensive - reuse for all chunks)
            s_tensor = agent.process_image(img_path, crop)
            
            # Pre-compute renderer encodings (reused for all chunks)
            with torch.no_grad():
                f_r, g_r = agent.renderer.dense_feature_encoder(s_tensor)
                t_lat = agent.renderer.latent_token_encoder(s_tensor)
                if isinstance(t_lat, tuple):
                    t_lat = t_lat[0]
            
            # Split audio into chunks
            chunk_paths = split_audio_into_chunks(aud_path, chunk_duration, chunk_dir)
            
            if not chunk_paths:
                raise ValueError("No audio chunks generated")
            
            # Define boundary for multipart response
            boundary = b"--CHUNK_BOUNDARY--"
            
            # Generate video for each audio chunk
            for i, chunk_path in enumerate(chunk_paths):
                print(f"Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
                
                # Generate video bytes for this chunk
                video_bytes = agent.generate_chunk(
                    img_tensor=s_tensor,
                    aud_path=chunk_path,
                    f_r=f_r,
                    g_r=g_r,
                    t_lat=t_lat,
                    cfg_scale=cfg_scale,
                    nfe=nfe
                )
                
                # Yield chunk with boundary marker and metadata
                chunk_header = f"chunk_index:{i}\nchunk_count:{len(chunk_paths)}\ncontent_length:{len(video_bytes)}\n".encode()
                yield boundary + b"\n" + chunk_header + b"\n" + video_bytes + b"\n"
                
        except Exception as e:
            print(f"Streaming error: {e}")
            raise
        finally:
            # Cleanup
            cleanup_chunks(chunk_paths, chunk_dir)
            if os.path.exists(aud_path):
                os.unlink(aud_path)
    
    return StreamingResponse(
        chunk_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Chunk-Duration": str(chunk_duration),
            "X-Content-Type": "video/mp4",
            "X-Chunk-Boundary": "--CHUNK_BOUNDARY--"
        }
    )