import os
import tempfile
import subprocess
from typing import Optional, Literal
from functools import partial
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
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
torch.set_float32_matmul_precision('high')  # Enable TensorFloat32 for better performance
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(True)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(True)

from generator.FM import FMGenerator
from renderer.models import IMTRenderer
print = partial(print, flush=True)

app = FastAPI(title="IMTalker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==== TTS Configuration ====
def _read_secret(env_var: str, default: str = "") -> str:
    """Read secret from file path specified in env var, or return default."""
    path = os.environ.get(env_var)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return default


ELEVENLABS_API_KEY = _read_secret("ELEVENLABS_API_KEY_FILE")
ELEVENLABS_VOICE_ID = os.environ.get("VOICE_ID", "TX3LPaxmHKxFdv7VOQHJ")
DEFAULT_TTS_PREFERENCE = os.environ.get("TTS_PREFERENCE", "elevenlabs")


async def synthesize_elevenlabs(text: str, output_path: str, voice_id: Optional[str] = None) -> None:
    """Synthesize speech using ElevenLabs API."""
    print(f"[TTS] Starting ElevenLabs synthesis for {len(text)} chars...")
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
    
    vid = voice_id or ELEVENLABS_VOICE_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    print(f"[TTS] Using voice ID: {vid}")
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    print("[TTS] Calling ElevenLabs API...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"[TTS] ElevenLabs API error: {response.status_code}")
            raise HTTPException(
                status_code=502,
                detail=f"ElevenLabs API error: {response.text}"
            )
        print(f"[TTS] Received {len(response.content)} bytes from ElevenLabs")
        # ElevenLabs returns MP3, save and convert to WAV for consistency
        mp3_path = output_path.replace(".wav", ".mp3")
        with open(mp3_path, "wb") as f:
            f.write(response.content)
        print("[TTS] Converting MP3 to WAV...")
        # Convert MP3 to WAV using ffmpeg
        cmd = f"ffmpeg -i {mp3_path} -ar 16000 -ac 1 {output_path} -y -loglevel error"
        subprocess.call(cmd, shell=True)
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        print(f"[TTS] Audio saved to {output_path}")


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
        self.num_prev_frames = 2  # Reduced from 10 for faster processing
        # Optimized defaults
        self.ode_atol = 1e-6
        self.ode_rtol = 1e-6
        self.nfe = 7
        self.torchdiffeq_ode_method = 'euler'
        # CFG scale: 1.0 = no CFG (fastest), >1.0 = 2x ODE computation per step
        self.a_cfg_scale = 1.0  # Minimal movement for subtle lip-sync focused output
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
        
        # Apply torch.compile for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("Applying torch.compile to renderer (this may take a moment on first run)...")
            try:
                # Compile the hot path functions
                self.renderer.adapt = torch.compile(self.renderer.adapt, mode='default')
                self.renderer.latent_token_decoder = torch.compile(self.renderer.latent_token_decoder, mode='default')
                self.renderer.decode = torch.compile(self.renderer.decode, mode='default')
                print("torch.compile applied successfully")
            except Exception as e:
                print(f"torch.compile failed (will use eager mode): {e}")
        
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
        print(f"[Image] Loading image from {img_path}...")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        print(f"[Image] Original size: {img.shape[1]}x{img.shape[0]}")
        
        if crop:
            print("[Image] Detecting face for cropping...")
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
                print(f"[Image] Cropped to {side}x{side}")
            else:
                print("[Image] No face detected, using full image")
        
        print("[Image] Transforming and moving to device...")
        return self.transform(img_pil).unsqueeze(0).to(self.device, non_blocking=True)
    
    def process_audio(self, aud_path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        print(f"[Audio] Loading audio from {aud_path}...")
        speech_array, sr = librosa.load(aud_path, sr=self.opt.sampling_rate)
        duration = len(speech_array) / sr
        print(f"[Audio] Duration: {duration:.2f}s, Sample rate: {sr}Hz")
        print("[Audio] Processing with Wav2Vec2...")
        result = self.wav2vec_preprocessor(
            speech_array, sampling_rate=sr, return_tensors='pt'
        ).input_values[0].unsqueeze(0).to(self.device, non_blocking=True)
        print("[Audio] Audio preprocessed and moved to device")
        return result
    
    @torch.no_grad()
    def generate(self, img_path: str, aud_path: str, output_path: str, 
                 crop: bool = True, cfg_scale: float = 3.0, nfe: int = 7) -> str:
        print(f"\n[Generate] Starting generation (cfg_scale={cfg_scale}, nfe={nfe})")
        
        # Preprocess inputs
        print("[Generate] Step 1/6: Processing image...")
        s_tensor = self.process_image(img_path, crop)
        print("[Generate] Step 2/6: Processing audio...")
        a_tensor = self.process_audio(aud_path)
        
        # Encode source image (done once, reused for all frames)
        print("[Generate] Step 3/6: Encoding source image...")
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        print("[Generate] Source image encoded")
        
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
        print(f"[Generate] Step 4/6: Generating motion latents cfg_scale: {cfg_scale}, nfe: {nfe}...")
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=self.opt.seed)
        print(f"[Generate] Generated {sample.shape[1]} motion frames")
        
        # Decode to frames - simple loop (batching doesn't help here due to memory constraints)
        T = sample.shape[1]
        print(f"[Generate] Step 5/6: Rendering {T} frames...")
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        d_hat = []
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for t in range(T):
                # Mark step boundary for CUDA graphs (required with torch.compile reduce-overhead)
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                d_hat.append(self.renderer.decode(m_c, m_r, f_r))
                if (t + 1) % 25 == 0 or t == T - 1:
                    print(f"[Generate] Rendered frame {t + 1}/{T}")
        
        vid_tensor = torch.stack(d_hat, dim=1).squeeze()
        print(f"[Generate] All frames rendered, tensor shape: {vid_tensor.shape}")
        
        # Ensure CUDA operations complete before video save
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Save video (resize on GPU if needed)
        print("[Generate] Step 6/6: Saving video...")
        return self._save_video(vid_tensor, output_path, aud_path)
    
    def _save_video(self, vid_tensor, output_path, audio_path):
        """Save video with audio, resizing on GPU if needed.
        
        Args:
            output_size: Target size for output video (1:1 ratio)
        """
        print("[Save] Preparing video tensor...")
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            temp_path = tmp.name
        
        vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
        vid = (vid * 255).type(torch.ByteTensor)
        print(f"[Save] Writing {vid.shape[0]} frames to temp file...")
        torchvision.io.write_video(temp_path, vid, fps=self.opt.fps)
        
        if audio_path:
            print("[Save] Muxing audio with video...")
            cmd = f'ffmpeg -i {temp_path} -i {audio_path} -c:v copy -c:a aac {output_path} -y -loglevel error'
            subprocess.call(cmd, shell=True)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            import shutil
            shutil.move(temp_path, output_path)
        
        print(f"[Save] Video saved to {output_path}")
        return output_path


# Initialize agent once at startup (models loaded here)
print("Initializing IMTalker inference agent...")
config = InferenceConfig()
agent = InferenceAgent(config)


@app.post("/generate")
async def generate_video(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    user_id: str = Form(...),
    avatar: str = Form(...),
    crop: bool = Form(True),
    cfg_scale: float = Form(1.0),
    nfe: int = Form(7),
    tts_preference: Optional[Literal["elevenlabs", "coqui"]] = Form(None, description="TTS provider: 'elevenlabs' or 'coqui'"),
    reference_aud_url: Optional[str] = Form(None),
    clone: Optional[str] = Form(None),
    split_sentences: bool = Form(False),
    speed: float = Form(1.0),
    voice_id: Optional[str] = Form(None, description="ElevenLabs voice ID (uses default if not provided)")
):
    """Generate talking face video from audio or text using default avatar.
    
    Provide either 'audio' (uploaded file) or 'text' (for TTS synthesis).
    When 'text' is provided, the TTS service at tts:8000/generate is called.
    
    This endpoint uses a persistent model - no subprocess overhead.
    """
    
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either 'audio' or 'text' must be provided")
    
    img_path = f"/app/user_img/{avatar}.jpg"
    if not os.path.exists(img_path):
        img_path = "/app/img/avatar_chest.jpg"

    output_dir = f"/app/results/{user_id}/"
    os.makedirs(output_dir, exist_ok=True)
    
    aud_path = f"/app/aud/{user_id}_audio.wav"
    os.makedirs(os.path.dirname(aud_path), exist_ok=True)
    
    try:
        print(f"\n[API] === New request from user_id={user_id} ===")
        if text:
            # Determine TTS provider
            provider = tts_preference or DEFAULT_TTS_PREFERENCE
            print(f"[API] Text provided ({len(text)} chars), using TTS provider: {provider}")
            
            if provider == "elevenlabs":
                # Use ElevenLabs API
                await synthesize_elevenlabs(text, aud_path, voice_id)
            else:
                # Use Coqui TTS service
                print("[API] Calling Coqui TTS service...")
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
                        print(f"[API] Coqui TTS error: {tts_response.status_code}")
                        raise HTTPException(
                            status_code=502, 
                            detail=f"TTS service error: {tts_response.text}"
                        )
                    print(f"[API] Received {len(tts_response.content)} bytes from Coqui TTS")
                    # Save the synthesized audio
                    with open(aud_path, "wb") as f:
                        f.write(tts_response.content)
        else:
            # Use uploaded audio file
            print("[API] Using uploaded audio file")
            with open(aud_path, "wb") as f:
                content = await audio.read()
                f.write(content)
            print(f"[API] Uploaded audio saved ({len(content)} bytes)")
            
        print(f"[API] Audio ready at {aud_path}")
        
        output_path = os.path.join(output_dir, f"{user_id}.mp4")
        
        # Direct inference - no subprocess!
        agent.generate(
            img_path=img_path,
            aud_path=aud_path,
            output_path=output_path,
            crop=crop,
            cfg_scale=cfg_scale,
            nfe=nfe
        )
        
        print("[API] Generation complete!")
        
        # Clean up temp audio
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        
        print(f"[API] Returning video: {output_path}")
        print(f"[API] === Request complete for user_id={user_id} ===\n")
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