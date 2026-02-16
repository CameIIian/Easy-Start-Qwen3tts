# Easy-Start-Qwen3tts

`uv` 環境で `Qwen/Qwen3-TTS-12Hz-1.7B-Base` をすぐに試すための最小構成です。

## 前提

- Python 3.10+
- `uv`（https://docs.astral.sh/uv/）
- （推奨）CUDA が使える GPU

## セットアップ

```bash
uv sync
```

## 実行例

```bash
uv run qwen3tts "こんにちは。Qwen3-TTSの音声テストです。" -o output.wav
```

オプション例:

```bash
uv run qwen3tts "This is an English sample." --voice Chelsie --max-new-tokens 2048 -o sample.wav
```

## 主要オプション

- `--model`: 既定値は `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `--voice`: 話者プリセット（既定値: `Chelsie`）
- `--thinker`: thinker モードを有効化
- `--max-new-tokens`: 生成トークン上限

## 補足

- 初回実行時にモデルがダウンロードされるため時間がかかります。
- CPU でも動作しますが非常に遅くなる可能性があります。
