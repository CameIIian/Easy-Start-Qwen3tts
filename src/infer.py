import torch    # qwen3tts
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from datetime import datetime   # filename
import os

def qwen3ttf(ref_audio, ref_text, inf_text):
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    wavs, sr = model.generate_voice_clone(
        text=inf_text,
        language="Japanese",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"./outputs/Qwen3tts-Base_{timestamp}.wav"
    sf.write(filename, wavs[0], sr)
    print(f"Saved: {filename}")

if "__main__" in __name__:
    for _ in range(1):
        ref_audio = "./inputs/voice.wav"
        ref_text =  ""
        inf_text =  ""
        qwen3ttf(ref_audio, ref_text, inf_text)
