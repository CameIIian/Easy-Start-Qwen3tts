# Easy-Start-Qwen3tts

`uv` 環境で `Qwen/Qwen3-TTS-12Hz-1.7B-Base` を実行するためのプロジェクト。

## TO-DO
[x] Qwen3ttsを動作させる\
[ ] FlashAttensionを追加する

## 前提
- `uv`（https://docs.astral.sh/uv/）
- Nvidiaドライバ / nvcc (Nvidia環境のみ)
- sox

## セットアップ
```sh
uv sync
```

## 実行例
起動チェック
```sh
uv run src/check.py
```

推論
```sh
uv run src/infer.py
```
