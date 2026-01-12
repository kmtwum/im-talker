import os
import tempfile
import subprocess
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
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
        self.num_prev_frames = 10
        # Optimized defaults
        self.ode_atol = 1e-4
        self.ode_rtol = 1e-4
        self.nfe = 7
        self.torchdiffeq_ode_method = 'euler'
        self.a_cfg_scale = 3.0
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
        
        print("Inference agent ready!")
    
    def _load_renderer(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_dict = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
        self.renderer.load_state_dict(clean_dict, strict=False)
    
    def _load_generator(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        prefix = 'model.'
        clean_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        with torch.no_grad():
            for name, param in self.generator.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name].to(self.device))
    
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
        
        return self.transform(img_pil).unsqueeze(0).to(self.device)
    
    def process_audio(self, aud_path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        speech_array, sr = librosa.load(aud_path, sr=self.opt.sampling_rate)
        return self.wav2vec_preprocessor(
            speech_array, sampling_rate=sr, return_tensors='pt'
        ).input_values[0].unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def generate(self, img_path: str, aud_path: str, output_path: str, 
                 crop: bool = True, cfg_scale: float = 3.0, nfe: int = 7) -> str:
        """Run inference and return video path"""
        
        # Preprocess inputs
        s_tensor = self.process_image(img_path, crop)
        a_tensor = self.process_audio(aud_path)
        
        # Encode source image
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
        
        # Decode to frames (batched for speed)
        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        d_hat = []
        batch_size = 4
        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)
            for t in range(t_start, t_end):
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                d_hat.append(self.renderer.decode(m_c, m_r, f_r))
        
        vid_tensor = torch.stack(d_hat, dim=1).squeeze()
        
        # Save video
        return self._save_video(vid_tensor, output_path, aud_path)
    
    def _save_video(self, vid_tensor, output_path, audio_path):
        """Save video with audio"""
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
            os.rename(temp_path, output_path)
        
        return output_path


# Initialize agent once at startup (models loaded here)
print("Initializing IMTalker inference agent...")
config = InferenceConfig()
agent = InferenceAgent(config)


@app.post("/generate")
async def generate_video(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
    crop: bool = Form(True),
    cfg_scale: float = Form(3.0),
    nfe: int = Form(7)
):
    """Generate talking face video from audio using default avatar.
    
    This endpoint uses a persistent model - no subprocess overhead.
    """
    
    img_path = "/app/img/avatar.jpg"
    output_dir = f"/app/results/{user_id}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save uploaded audio
    aud_path = f"/app/aud/{user_id}_audio.wav"
    os.makedirs(os.path.dirname(aud_path), exist_ok=True)
    with open(aud_path, "wb") as f:
        content = await audio.read()
        f.write(content)
    
    try:
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
        
        # Clean up temp audio
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"generated_{user_id}.mp4"
        )
        
    except Exception as e:
        if os.path.exists(aud_path):
            os.unlink(aud_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": agent is not None}