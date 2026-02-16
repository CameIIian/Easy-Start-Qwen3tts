# Easy-Start-Qwen3tts

`pyenv` 環境で `Qwen/Qwen3-TTS-12Hz-1.7B-Base` を実行するためのプロジェクト\
`uv`でのトラブルが解決できたら移行予定

## TO-DO
[x] Qwen3ttsを動作させる\
[ ] FlashAttensionを追加する

## 前提
- [pyenv](https://github.com/pyenv/pyenv)
- [Nvidiaドライバ](https://github.com/canonical/ubuntu-drivers-common) / [nvcc](https://developer.nvidia.com/cuda-12-4-0-download-archive) (Nvidia環境のみ, cuda>=11.8)
- [sox](https://en.ubunlog.com/sox-plays-mp3-terminal/)

## セットアップ
```sh
pyenv install 3.12.11
pyenv local 3.12.11
pip install -r requirements.txt
```

## 実行例
起動チェック
```sh
python src/check.py
```

推論
```sh
python src/infer.py
```

学習 (wip, unslothが対応後に作成)
```sh
python src/infer.py
```

## 結果
Program: 実行されたプログラム\
wav_length: 入力音源の長さ\
text_type: 全て平仮名(以下**かな**) または 平仮名+カタカナ+漢字(以下**漢字**)\
vram: 使用量, qwen3ttsが専有している部分のみ, GB\
安定性: 出力された音が安定しているか, 定性的評価\

### モデル読み込み
| Program | vram |
| --- | --- |
| check.py | 4.0 |

### 推論(出力: 仮名)
| Program | wav_length | text_style | vram | 安定性 |
| --- | --- | --- | --- | --- |
| infer.py | in: 2s | かな | 4.5 | ▲ |
| infer.py | in: 2s | 漢字 | 4.5 | × |
| infer.py | in: 11s | 漢字 | 5.0 | ○ |

### 推論(出力: 漢字)
| Program | wav_length | text_style | vram | 安定性 |
| --- | --- | --- | --- | --- |
| infer.py | in: 2s | かな | 4.5 | ○ |
| infer.py | in: 2s | 漢字 | 4.5 | ▲ |
| infer.py | in: 11s | 漢字 | 5.0 | ✅ |

### 学習
| Program | vram |
| --- | --- |
| learn.py | --- |

## 考察/注意点
意外と平仮名in/outよりも、漢字in/outの方が良さげ\
文脈を捉えるまでは出来てないものの、単語は捉えられている

入力音声の影響か、所々全体的にノイズが乗っていた\
高品質データが必須

## トラブルシューティング
### flash attention のインストールが終わらない
下に無い組み合わせは自前でbuildしようとするので劇遅
> https://github.com/Dao-AILab/flash-attention/releases

### uv + torch==2.9.0 + flash-attn だとPreparingから進まない
上手くPackageを追加できてない？\
要分析
