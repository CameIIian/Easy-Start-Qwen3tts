import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_audio = "./inputs/voice.wav"
ref_text  = ""

wavs, sr = model.generate_voice_clone(
    text="",
    language="Japanese",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("./outputs/output_voice_clone.wav", wavs[0], sr)