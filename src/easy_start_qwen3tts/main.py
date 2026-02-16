from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen3TTSForConditionalGeneration

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def synthesize(
    text: str,
    output_path: Path,
    model_id: str,
    voice: str,
    max_new_tokens: int,
    thinker: bool,
) -> None:
    """Generate speech audio from text using Qwen3-TTS."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            speaker=voice,
            thinker=thinker,
            max_new_tokens=max_new_tokens,
        )

    audio = generated[0].detach().float().cpu().numpy()
    sample_rate = getattr(model.config, "sampling_rate", 24000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS from a uv-managed environment")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.wav"), help="Output WAV path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--voice", default="Chelsie", help="Voice preset")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Generation length")
    parser.add_argument("--thinker", action="store_true", help="Enable thinker mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesize(
        text=args.text,
        output_path=args.output,
        model_id=args.model,
        voice=args.voice,
        max_new_tokens=args.max_new_tokens,
        thinker=args.thinker,
    )
    print(f"Saved audio to {args.output}")


if __name__ == "__main__":
    main()
