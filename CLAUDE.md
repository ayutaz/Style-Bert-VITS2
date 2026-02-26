# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Style-Bert-VITS2は、Bert-VITS2 v2.1をベースにした日本語/多言語対応のText-to-Speech (TTS) システム。感情や発話スタイルを連続的に制御できる点が特徴。Python >=3.9、ライセンスはAGPL-3.0。

- Pythonライブラリ (`pip install style-bert-vits2`) としても、WebUI/APIサーバーとしても利用可能
- 対応言語: JP (日本語), EN (英語), ZH (中国語)
- JP-Extraモデル: 日本語特化の高品質モデルバリアント（JP以外の言語は使用不可）

## よく使うコマンド

### 環境構築
```bash
uv venv venv && venv\Scripts\activate
uv pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
python initialize.py  # BERTモデル・デフォルトTTSモデルのダウンロード
```

### 起動
```bash
python app.py                        # WebUI (Gradio、6タブ構成)
python app.py --device cpu           # CPUモード
python server_fastapi.py             # FastAPI サーバー (port 5000)
python server_editor.py --inbrowser  # エディターUI
```

### テスト
```bash
# hatch経由
hatch run test:test                  # PyTorch CPU テスト
hatch run test:test-cuda             # PyTorch CUDA テスト
hatch run test-onnx:test             # ONNX CPU テスト
hatch run test-onnx:test-directml   # ONNX DirectML テスト (Windows)

# 直接実行（単一テスト）
pytest -s tests/test_main.py::test_synthesize_cpu
pytest -s tests/test_main.py::test_synthesize_onnx_cuda
```

### コードスタイル
```bash
hatch run style:check   # black + isort チェック
hatch run style:fmt     # black + isort 自動修正
```

### 学習パイプライン (CLI)
```bash
python slice.py --model_name <name>              # 音声スライス
python transcribe.py --model_name <name>         # 書き起こし
python preprocess_all.py -m <name> [--use_jp_extra]  # 前処理一括
python train_ms.py                               # 通常モデル学習
python train_ms_jp_extra.py                      # JP-Extraモデル学習
```

### その他
```bash
python convert_onnx.py          # ONNX変換
python speech_mos.py -m <name>  # 自然性評価 (SpeechMOS)
```

## アーキテクチャ

### コアパッケージ: `style_bert_vits2/`

ライブラリとしてpip公開されている部分。推論のコアロジックが全てここに集約されている。

- **`tts_model.py`** — 推論のメインエントリポイント
  - `TTSModel`: 単一モデルの読み込み・推論・アンロード。safetensorsとONNXの両方に対応
  - `TTSModelHolder`: 複数モデルの管理。`model_assets/` 配下のモデルを自動検出
- **`models/`** — ニューラルネットワーク本体
  - `models.py`: 標準VITS2アーキテクチャ (`SynthesizerTrn`, `MultiPeriodDiscriminator`)
  - `models_jp_extra.py`: JP-Extra版 (WavLMベースの判別器を追加)
  - `infer.py`: モデルバリアント選択ヘルパー `get_net_g()`
  - `hyper_parameters.py`: Pydanticベースの設定モデル
- **`nlp/`** — テキスト処理 (言語ごとにサブディレクトリ)
  - `__init__.py`: 言語非依存API (`extract_bert_feature()`, `clean_text()`, `cleaned_text_to_sequence()`)
  - `japanese/`, `english/`, `chinese/`: 各言語のG2P、BERT特徴抽出、正規化
  - `japanese/pyopenjtalk_worker/`: GIL回避用のTCPソケットサーバーパターン
  - `japanese/user_dict/`: VOICEVOXベースのユーザー辞書 (LGPL v3)
- **`constants.py`** — バージョン (`VERSION`)、デフォルトパラメータ、言語定義。hatchのバージョンソースでもある

### 推論データフロー

```
テキスト入力
  → 言語別テキスト正規化 (nlp/{lang}/normalizer.py)
  → G2P変換 (nlp/{lang}/g2p.py) → phones, tones, word2ph
  → BERT特徴抽出 (nlp/{lang}/bert_feature.py) — 言語別BERTモデル使用
  → シンボルID変換 (cleaned_text_to_sequence)
  → SynthesizerTrn.infer() — エンコーダ→デュレーション予測→デコーダ→フロー→ボコーダ
  → スタイルベクトル適用 (style_vectors.npy から読み込み)
  → 音声波形出力
```

### WebUI: `gradio_tabs/`

`app.py` がGradioアプリのエントリポイント。6つのタブ:
- `inference.py`: 音声合成
- `dataset.py`: データセット作成
- `train.py`: 学習
- `style_vectors.py`: スタイル生成
- `merge.py`: モデルマージ
- `convert_onnx.py`: ONNX変換

### APIサーバー

- `server_fastapi.py`: FastAPIベースのREST API (ポート5000)。`/docs`でSwagger UI
- `server_editor.py`: エディター専用API

### 学習関連 (トップレベルスクリプト)

- `train_ms.py` / `train_ms_jp_extra.py`: 学習メインスクリプト
- `preprocess_all.py`: 前処理パイプライン一括実行
- `bert_gen.py`: BERT特徴抽出
- `style_gen.py` / `default_style.py`: スタイルベクトル生成
- `data_utils.py`: データセットローダー

### 設定ファイル

- `configs/config.json` / `config_jp_extra.json`: モデルハイパーパラメータテンプレート
- `configs/default_paths.yml`: データセット・アセットのルートパス定義
- `config.yml` (実行時生成): `default_config.yml` から生成されるユーザー設定

### モデルアセット構造

```
model_assets/{model_name}/
  ├── config.json          # モデル設定
  ├── *.safetensors        # モデル重み
  └── style_vectors.npy    # スタイルベクトル (256次元, wespeaker埋め込み)
```

## 重要な設計判断

- モデルは遅延ロード・明示的アンロード方式 (VRAM管理のため)
- 日本語テキスト処理はpyopenjtalkのGIL問題を回避するため、別プロセスでTCPソケットサーバーとして動作
- スタイル制御は256次元のwespeaker話者埋め込みベースで、`style_weight`パラメータで効果の強さを連続的に調整可能
- safetensors形式がデフォルトのモデル保存形式
- ONNX推論はPyTorch非依存で動作可能（DirectML/CoreML対応）
