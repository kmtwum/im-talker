import glob
import os
import tempfile
import subprocess
import uuid
from typing import Optional, Literal, List, AsyncGenerator
from functools import partial
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
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

# ==== fMP4 Streaming Codec Configuration ====
# Using H.265 (HEVC) + Opus for better compression in real-time streaming
# VIDEO_CODEC = "libx265"
# VIDEO_CODEC_PARAMS = "-preset ultrafast -tune zerolatency"
# VIDEO_CODEC_STRING = "hvc1.1.6.L93.B0"  # For MediaSource mime type
# AUDIO_CODEC = "libopus"
# AUDIO_CODEC_PARAMS = "-b:a 64k"
# AUDIO_CODEC_STRING = "opus"  # For MediaSource mime type

# Using H.264 + AAC for broader browser compatibility
VIDEO_CODEC = "libx264"
VIDEO_CODEC_PARAMS = "-preset ultrafast -tune zerolatency"
VIDEO_CODEC_STRING = "avc1.42E01E"  # For MediaSource mime type
AUDIO_CODEC = "aac"
AUDIO_CODEC_PARAMS = "-b:a 128k"
AUDIO_CODEC_STRING = "mp4a.40.2"  # For MediaSource mime type

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


async def synthesize_coqui(
    text: str,
    output_path: str,
    reference_aud_url: Optional[str] = None,
    clone: Optional[str] = None,
    split_sentences: bool = False,
    speed: float = 1.0
) -> None:
    """Synthesize speech using Coqui TTS service."""
    print(f"[TTS] Starting Coqui synthesis for {len(text)} chars...")
    
    tts_payload = {
        "text": text,
        "source_aud": reference_aud_url or "",
        "split_sentences": split_sentences,
        "streaming": False,
        "speed": speed
    }
    if clone:
        tts_payload["clone"] = clone
    
    print("[TTS] Calling Coqui TTS service...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://tts:8000/generate",
            json=tts_payload
        )
        if response.status_code != 200:
            print(f"[TTS] Coqui TTS error: {response.status_code}")
            raise HTTPException(
                status_code=502,
                detail=f"TTS service error: {response.text}"
            )
        print(f"[TTS] Received {len(response.content)} bytes from Coqui TTS")
        with open(output_path, "wb") as f:
            f.write(response.content)
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
        self.fps = 15.0
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

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self._load_avatars()

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

    def _load_avatars(self):
        self.avatars = ['sunny', 'jamal']
        self.avatar_pils = {}

        for avatar in self.avatars:
            img = cv2.imread(f"/app/user_img/{avatar}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            self.avatar_pils[avatar] = self.transform(img_pil).unsqueeze(0).to(self.device, non_blocking=True)

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
    def generate(self, avatar: str, aud_path: str, output_path: str, cfg_scale: float = 3.0, nfe: int = 7) -> str:
        print(f"\n[Generate] Starting generation (cfg_scale={cfg_scale}, nfe={nfe})")
        
        # Preprocess inputs
        print("[Generate] Step 1/6: Processing image...")
        s_tensor = self.avatar_pils[avatar]
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
                frame = self.renderer.decode(m_c, m_r, f_r)
                # d_hat.append(frame) #GPU

                # Move to CPU immediately to avoid GPU OOM for long videos
                d_hat.append(frame.float().cpu())
                if (t + 1) % 25 == 0 or t == T - 1:
                    print(f"[Generate] Rendered frame {t + 1}/{T}")
        
        # Stack on CPU (already there)
        vid_tensor = torch.stack(d_hat, dim=1).squeeze(0)  # Remove batch dim only, keep time dim
        print(f"[Generate] All frames rendered, tensor shape: {vid_tensor.shape}")

        # No need to synchronize - frames already moved to CPU
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

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
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for t in range(T):
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                d_hat.append(self.renderer.decode(m_c, m_r, f_r))

        vid_tensor = torch.stack(d_hat, dim=1).squeeze(0)  # Remove batch dim only, keep time dim

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

    def _encode_to_fmp4_segment(self, vid_tensor: torch.Tensor, audio_path: str, 
                                 is_first: bool = False) -> tuple:
        """Encode video tensor to fMP4 segment for MediaSource streaming.
        
        Uses fragmented MP4 format with movflags for streaming compatibility.
        
        Args:
            vid_tensor: Video frames tensor [T, C, H, W]
            audio_path: Path to audio file for this segment
            is_first: If True, also extract and return initialization segment
            
        Returns:
            Tuple of (media_segment_bytes, init_segment_bytes or None)
        """
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_vid:
            temp_vid_path = tmp_vid.name
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            temp_out_path = tmp_out.name

        try:
            # Write video frames to temp file
            vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
            vid = (vid * 255).type(torch.ByteTensor)
            torchvision.io.write_video(temp_vid_path, vid, fps=self.opt.fps)

            # Encode to fMP4 with proper flags for MediaSource
            # -movflags frag_keyframe+empty_moov+default_base_moof enables:
            #   - frag_keyframe: Fragment at each keyframe
            #   - empty_moov: Put moov at start with no sample data (streaming)
            #   - default_base_moof: Use moof as base for offsets (MSE compatibility)
            # -force_key_frames ensures IDR frames at fragment boundaries
            cmd = [
                'ffmpeg', '-i', temp_vid_path, '-i', audio_path,
                '-c:v', VIDEO_CODEC] + VIDEO_CODEC_PARAMS.split() + [
                '-c:a', AUDIO_CODEC] + AUDIO_CODEC_PARAMS.split() + [
                '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
                '-force_key_frames', 'expr:gte(t,n_forced*1)',  # Force keyframe every second
                '-f', 'mp4',
                temp_out_path,
                '-y', '-loglevel', 'error'
            ]
            subprocess.call(cmd)

            # Read the fMP4 bytes
            with open(temp_out_path, 'rb') as f:
                fmp4_bytes = f.read()

            # Always separate Media Segment from Init Segment
            # Media segment is everything from first 'moof' onwards
            moof_pos = fmp4_bytes.find(b'moof')
            if moof_pos > 4:
                media_segment = fmp4_bytes[moof_pos - 4:]  # Include box size
            else:
                # Should not happen with frag_keyframe+empty_moov
                media_segment = fmp4_bytes

            if is_first:
                # Extract initialization segment (everything before first 'moof')
                init_segment = self._extract_init_segment(fmp4_bytes)
                return (media_segment, init_segment)
            else:
                # For subsequent chunks, ONLY return the media segment
                # (ffmpeg always generates a full file with headers, we must strip them)
                return (media_segment, None)

        finally:
            # Cleanup temp files
            if os.path.exists(temp_vid_path):
                os.remove(temp_vid_path)
            if os.path.exists(temp_out_path):
                os.remove(temp_out_path)

    def _extract_init_segment(self, fmp4_bytes: bytes) -> bytes:
        """Extract initialization segment from fMP4 data.
        
        The init segment contains ftyp + moov boxes with codec metadata
        needed by MediaSource before any media segments can be appended.
        
        Args:
            fmp4_bytes: Complete fMP4 file bytes
            
        Returns:
            Initialization segment bytes (ftyp + moov)
        """
        # Find 'moof' marker - init segment is everything before it
        moof_pos = fmp4_bytes.find(b'moof')
        if moof_pos == -1:
            raise ValueError("No moof box found - not a valid fMP4")
        
        # Init segment ends 4 bytes before 'moof' (the box size field)
        init_end = moof_pos - 4
        return fmp4_bytes[:init_end]


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

    request_id = str(uuid.uuid4())[:8]
    
    output_dir = f"/app/results/{user_id}/{request_id}/"
    os.makedirs(output_dir, exist_ok=True)
    
    aud_path = f"/app/aud/{user_id}_{request_id}_audio.wav"
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
                await synthesize_coqui(
                    text, aud_path,
                    reference_aud_url=reference_aud_url,
                    clone=clone,
                    split_sentences=split_sentences,
                    speed=speed
                )
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
            avatar=avatar,
            aud_path=aud_path,
            output_path=output_path,
            cfg_scale=cfg_scale,
            nfe=nfe
        )

        print("[API] Generation complete!")

        # Clean up temp audio
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        
        print(f"[API] Returning video: {output_path}")
        print(f"[API] === Request complete for user_id={user_id} ===\n")
        
        # Cleanup after response is sent
        def cleanup_output_dir():
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
                print(f"[API] Cleaned up {output_dir}")
        
        background_tasks = BackgroundTasks()
        background_tasks.add_task(cleanup_output_dir)
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"generated_{user_id}.mp4",
            background=background_tasks
        )
        
    except httpx.RequestError as e:
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=502, detail=f"Failed to connect to TTS service: {str(e)}")
    except Exception as e:
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": agent is not None}


@app.post("/generate/stream")
async def generate_video_stream(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    user_id: str = Form(...),
    avatar: str = Form(...),
    cfg_scale: float = Form(1.0),
    nfe: int = Form(7),
    chunk_duration: float = Form(5.0, ge=2.0, le=15.0, description="Duration of each video chunk in seconds"),
    tts_preference: Optional[Literal["elevenlabs", "coqui"]] = Form(None),
    voice_id: Optional[str] = Form(None),
    reference_aud_url: Optional[str] = Form(None),
    clone: Optional[str] = Form(None),
    split_sentences: bool = Form(False),
    speed: float = Form(1.0)
):
    """Stream video as fMP4 chunks for MediaSource Extensions playback.
    
    This endpoint streams fragmented MP4 segments compatible with the browser's
    MediaSource Extensions API. The first bytes are the initialization segment
    containing codec metadata, followed by media segments.
    
    Client should use:
    - MediaSource.isTypeSupported('video/mp4; codecs="hvc1.1.6.L93.B0, opus"')
    - Append init segment first, then media segments to SourceBuffer
    
    Provide either 'audio' (uploaded file) or 'text' (for TTS synthesis).
    """
    
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either 'audio' or 'text' must be provided")

    # Validate avatar exists
    if avatar not in agent.avatars:
        raise HTTPException(status_code=400, detail=f"Avatar '{avatar}' not found. Available: {agent.avatars}")
    
    # Generate unique request ID to avoid conflicts from parallel requests
    request_id = str(uuid.uuid4())[:8]
    
    aud_path = f"/app/aud/{user_id}_{request_id}_stream_audio.wav"
    chunk_dir = f"/app/aud/{user_id}_{request_id}_chunks/"
    os.makedirs(os.path.dirname(aud_path), exist_ok=True)
    
    # Get full audio first (from upload or TTS)
    try:
        print(f"\n[Stream] === New streaming request from user_id={user_id} ===")
        if text:
            provider = tts_preference or DEFAULT_TTS_PREFERENCE
            print(f"[Stream] Text provided ({len(text)} chars), using TTS: {provider}")
            
            if provider == "elevenlabs":
                await synthesize_elevenlabs(text, aud_path, voice_id)
            else:
                await synthesize_coqui(
                    text, aud_path,
                    reference_aud_url=reference_aud_url,
                    clone=clone,
                    split_sentences=split_sentences,
                    speed=speed
                )
        else:
            print("[Stream] Using uploaded audio file")
            with open(aud_path, "wb") as f:
                content = await audio.read()
                f.write(content)
    except httpx.RequestError as e:
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        raise HTTPException(status_code=502, detail=f"TTS connection failed: {str(e)}")
    
    async def fmp4_chunk_generator() -> AsyncGenerator[bytes, None]:
        """Generate and yield fMP4 segments for MediaSource streaming."""
        chunk_paths = []
        try:
            # Get pre-loaded avatar tensor
            s_tensor = agent.avatar_pils[avatar]
            
            # Pre-compute renderer encodings (reused for all chunks)
            print("[Stream] Pre-computing renderer encodings...")
            with torch.no_grad():
                f_r, g_r = agent.renderer.dense_feature_encoder(s_tensor)
                t_lat = agent.renderer.latent_token_encoder(s_tensor)
                if isinstance(t_lat, tuple):
                    t_lat = t_lat[0]
            
            # Split audio into chunks
            print(f"[Stream] Splitting audio into {chunk_duration}s chunks...")
            chunk_paths = split_audio_into_chunks(aud_path, chunk_duration, chunk_dir)
            
            if not chunk_paths:
                raise ValueError("No audio chunks generated")
            
            print(f"[Stream] Processing {len(chunk_paths)} chunks...")
            
            # Generate video for each audio chunk
            for i, chunk_path in enumerate(chunk_paths):
                is_first = (i == 0)
                print(f"[Stream] Generating chunk {i+1}/{len(chunk_paths)}...")
                
                # Wrap entire chunk generation in no_grad to prevent OOM
                with torch.no_grad():
                    # Process audio for this chunk
                    a_tensor = agent.process_audio(chunk_path)
                    
                    # Generate motion latents
                    data = {
                        's': s_tensor,
                        'a': a_tensor,
                        'pose': None,
                        'cam': None,
                        'gaze': None,
                        'ref_x': t_lat
                    }
                    sample = agent.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=agent.opt.seed)
                    
                    # Free audio tensor immediately
                    del a_tensor
                    
                    # Render frames
                    T = sample.shape[1]
                    ta_r = agent.renderer.adapt(t_lat, g_r)
                    m_r = agent.renderer.latent_token_decoder(ta_r)
                    
                    d_hat = []
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        for t in range(T):
                            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                                torch.compiler.cudagraph_mark_step_begin()
                            ta_c = agent.renderer.adapt(sample[:, t, ...], g_r)
                            m_c = agent.renderer.latent_token_decoder(ta_c)
                            frame = agent.renderer.decode(m_c, m_r, f_r)
                            d_hat.append(frame.float().cpu())
                    
                    # Free sample tensor
                    del sample
                    
                    vid_tensor = torch.stack(d_hat, dim=1).squeeze(0)  # Remove batch dim only, keep time dim
                    del d_hat
                
                # Encode to fMP4 segment (vid_tensor is on CPU, safe outside no_grad)
                media_bytes, init_bytes = agent._encode_to_fmp4_segment(
                    vid_tensor, chunk_path, is_first=is_first
                )
                
                # Free vid_tensor and clear GPU cache
                del vid_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Yield init segment first (only for first chunk)
                if is_first and init_bytes:
                    print(f"[Stream] Sending init segment ({len(init_bytes)} bytes)")
                    yield init_bytes
                
                # Yield media segment
                print(f"[Stream] Sending media segment {i+1} ({len(media_bytes)} bytes)")
                yield media_bytes
                
            print("[Stream] All chunks sent!")
                
        except Exception as e:
            print(f"[Stream] Error: {e}")
            raise
        finally:
            # Cleanup
            cleanup_chunks(chunk_paths, chunk_dir)
            if os.path.exists(aud_path):
                os.unlink(aud_path)
            print(f"[Stream] === Stream complete for user_id={user_id} ===\n")
    
    # Return codec info in headers for client setup
    codec_string = f'video/mp4; codecs="{VIDEO_CODEC_STRING}, {AUDIO_CODEC_STRING}"'
    
    return StreamingResponse(
        fmp4_chunk_generator(),
        media_type="video/mp4",
        headers={
            "X-Codec-String": codec_string,
            "X-Chunk-Duration": str(chunk_duration),
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        }
    )