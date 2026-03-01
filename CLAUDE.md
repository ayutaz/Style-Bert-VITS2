# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Style-Bert-VITS2は、Bert-VITS2 v2.1をベースにした日本語/多言語対応のText-to-Speech (TTS) システム。感情や発話スタイルを連続的に制御できる点が特徴。Python 3.12（`.python-version`で固定）、ライセンスはAGPL-3.0。

- Pythonライブラリ (`pip install style-bert-vits2`) としても、WebUIとしても利用可能
- 対応言語: JP (日本語), EN (英語), ZH (中国語)
- JP-Extraモデル: 日本語特化の高品質モデルバリアント（JP以外の言語は使用不可）
- **このブランチは推論専用**: 学習・前処理・データセット作成機能は削除済み
- HTMXベースのモーフィングUI (`/morphing`) でスタイル間の補間操作が可能
- HTMXベースのベクトル演算UI (`/vector-arithmetic`) でスタイルベクトルの算術操作が可能

## よく使うコマンド

### 環境構築
```bash
uv sync --group infer                # 推論環境
uv sync --only-group style           # スタイルチェックのみ
python initialize.py                  # BERTモデル・デフォルトTTSモデルのダウンロード
```

### 起動
```bash
uv run python app.py                        # WebUI (FastAPI + HTMX)
uv run python app.py --device cpu           # CPUモード
```

### テスト
```bash
uv run --only-group test pytest tests/test_style_ops.py -v  # style_ops 単体テスト (PyTorch不要)
uv run --only-group test pytest tests/test_expression_parser.py -v  # 式パーサーテスト (PyTorch不要)
uv run --only-group test pytest tests/test_edge_cases.py -v        # エッジケーステスト (PyTorch不要)
uv run --only-group test pytest tests/test_integration.py -v       # 統合テスト (PyTorch不要)
uv run pytest -s tests/test_main.py::test_synthesize_cpu     # 音声合成CPUテスト
uv run pytest -s tests/test_main.py::test_synthesize_cuda    # 音声合成CUDAテスト
```

### コードスタイル
```bash
uv run ruff check .                                       # リントチェック
uv run ruff format --check .                              # フォーマットチェック
uv run ruff check --fix . && uv run ruff format .         # 自動修正
```

## アーキテクチャ

### コアパッケージ: `style_bert_vits2/`

ライブラリとしてpip公開されている部分。推論のコアロジックが全てここに集約されている。

- **`tts_model.py`** — 推論のメインエントリポイント
  - `TTSModel`: 単一モデルの読み込み・推論・アンロード。safetensorsとONNXの両方に対応
  - `TTSModelHolder`: 複数モデルの管理。`model_assets/` 配下のモデルを自動検出
  - `style_vector_override`: 推論時にスタイルベクトルを外部から直接指定するパラメータ（モーフィングUIで使用）
- **`models/`** — ニューラルネットワーク本体
  - `models.py`: 標準VITS2アーキテクチャ (`SynthesizerTrn`)
  - `models_jp_extra.py`: JP-Extra版
  - `infer.py`: モデルバリアント選択ヘルパー `get_net_g()`
  - `hyper_parameters.py`: Pydanticベースの設定モデル
- **`nlp/`** — テキスト処理 (言語ごとにサブディレクトリ)
  - `__init__.py`: 言語非依存API (`extract_bert_feature()`, `clean_text()`, `cleaned_text_to_sequence()`)
  - `japanese/`, `english/`, `chinese/`: 各言語のG2P、BERT特徴抽出、正規化
  - `japanese/pyopenjtalk_worker/`: GIL回避用のTCPソケットサーバーパターン
  - `japanese/user_dict/`: VOICEVOXベースのユーザー辞書 (LGPL v3)
- **`style_ops.py`** — スタイルベクトル演算モジュール。モーフィングUI (`morphing_app.py`) とベクトル演算UI (`vector_app.py`) から呼び出される
  - 補間: `lerp()`, `slerp()` (LERPフォールバック付き球面線形補間)
  - ベクトル演算: `vector_add()`, `vector_sub()`, `vector_mean()`, `vector_scale()`, `vector_diff_transfer()`
  - 安全機構: `clip_norm()`, `compute_norm_ratio()`, `validate_style_vector()`
  - I/O: `load_style_vectors()`, `save_style_vectors()`
  - 可視化: `pca_project_2d()` (SVDベースPCA), `prepare_plot_data()` (座標正規化5-95範囲)
- **`expression_parser.py`** — カスタム式パーサー。スタイル名と四則演算の式を安全に解析（eval不使用）
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

### HTMX UI（全画面共通）

`app.py` がFastAPIアプリのエントリポイント。全UIはHTMXベースで統一。

- **`inference_app.py`** — FastAPI APIRouterで実装された推論UIバックエンド
  - `/inference`: メインページ（HTMXベース）
  - `/api/inference/*`: モデル選択・ロード、音声合成、モデル一覧更新のAPIエンドポイント
- **`morphing_app.py`** — FastAPI APIRouterで実装されたモーフィングUIバックエンド
  - `/morphing`: メインページ（HTMXベース）
  - `/api/morphing/*`: モデル選択、スタイル補間、音声合成、スタイル保存のAPIエンドポイント
- **`vector_app.py`** — FastAPI APIRouterで実装されたベクトル演算UIバックエンド
  - `/vector-arithmetic`: メインページ（HTMXベース）
  - `/api/vector/*`: モデル選択、ベクトル演算（差分転写・加重平均・スケーリング・カスタム式）、音声合成、スタイル保存のAPIエンドポイント
- **`templates/`** — Jinja2テンプレート
  - `inference.html`: 推論UIメインページ
  - `morphing.html`: モーフィングUIメインページ
  - `vector_arithmetic.html`: ベクトル演算UIメインページ
  - `partials/inference_options.html`: 推論用パラメータ設定フラグメント
  - `partials/style_options.html`: モーフィング用スタイル選択フラグメント
  - `partials/va_style_options.html`: ベクトル演算用モード別スタイル選択フラグメント
  - `partials/audio_player.html`: 音声プレーヤーフラグメント
  - `partials/norm_indicator.html`: ノルム安全インジケータ
  - `partials/style_plot.html`: PCA 2Dスタイルプロット (SVG)
- **`static/css/morphing.css`** — HTMX UI共通ダークテーマスタイルシート
- **`app.py`** — FastAPIアプリ。`/` は `/inference` にリダイレクト

### 設定ファイル

- `configs/default_paths.yml`: アセットのルートパス定義
- `config.py`: パス設定の読み込み (`get_path_config()`)

### モデルアセット構造

```
model_assets/{model_name}/
  ├── config.json          # モデル設定
  ├── *.safetensors        # モデル重み
  └── style_vectors.npy    # スタイルベクトル (256次元, wespeaker埋め込み)
```

## 重要な設計判断

- モデルは遅延ロード・明示的アンロード方式 (VRAM管理のため)
- 日本語テキスト処理はpyopenjtalk-plusを使用。GIL問題を回避するため別プロセスでTCPソケットサーバーとして動作
- スタイル制御は256次元のwespeaker話者埋め込みベースで、`style_weight`パラメータで効果の強さを連続的に調整可能
- safetensors形式がデフォルトのモデル保存形式
- ONNX推論はPyTorch非依存で動作可能（DirectML/CoreML対応）
