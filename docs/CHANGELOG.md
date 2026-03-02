# Changelog

## [Unreleased] — 推論専用ブランチ・HTMX UI統一

### 追加 (Added)

#### スタイルベクトル演算コアライブラリ (`style_bert_vits2/style_ops.py`)
- 補間関数: `slerp()` (球面線形補間), `lerp()` (線形補間)
- ベクトル演算: `vector_add()`, `vector_sub()`, `vector_mean()`, `vector_scale()`, `vector_diff_transfer()`
- 安全機構: `clip_norm()` (ノルムクリッピング), `compute_norm_ratio()`, `validate_style_vector()`
- I/Oユーティリティ: `load_style_vectors()`, `save_style_vectors()`
- ユニットテスト 43件 (`tests/test_style_ops.py`)

#### 推論パイプライン拡張 (`style_bert_vits2/tts_model.py`)
- `TTSModel.infer()` に `style_vector_override` パラメータ追加
- PyTorch / ONNX 両推論パスで動作
- 既存APIとの完全な後方互換性を保持

#### 音声合成UI (`/inference`)
- HTMXベースの音声合成UI（Gradio UIを完全に置き換え）
- `inference_app.py`: FastAPI APIRouterによるバックエンド実装
- モデル選択・ロード、スタイル指定、各種パラメータ調整、行分割合成
- アシストテキスト機能

#### モーフィングUI (`/morphing`)
- HTMXベースのスタンドアロンモーフィングUI
- SLERP / LERP によるスタイル間補間
- リアルタイムノルム安全インジケータ
- 音声プレビュー再生
- 補間結果のスタイル保存機能

#### ベクトル演算UI (`/vector-arithmetic`)
- HTMXベースのベクトル演算UI
- 4つの演算モード:
  - 差分転写: A + scale * (B - C)
  - 加重平均: w1*A + w2*B + w3*C (正規化オプション)
  - スケーリング: scale * A
  - カスタム式: `Happy - 0.3 * Neutral + 0.5 * Sad` 形式の自由記述
- カスタム式パーサー (`style_bert_vits2/expression_parser.py`, eval不使用)
- ノルムクリッピング設定 (トグル + 倍率)
- 2×2モードカードグリッド、ライブ数式表示、リアクティブノルム更新
- 式パーサーテスト 15件 + ベクトル演算テスト 13件

#### 統合・品質保証
- 2Dスタイルベクトル可視化 (PCA射影)
- エッジケーステスト追加 (27件: `tests/test_edge_cases.py`)
- 統合テスト追加 (8件: `tests/test_integration.py`)
- 各UIにインタラクティブ使い方ガイド追加
- タブ形式ナビゲーションで全3画面を統一的に切り替え可能

#### コード品質基盤
- black + isort から ruff に完全移行（リンター + フォーマッター統合）
- ruff ルールセット: E, F, W, I, UP, B, SIM, RUF
- ruff check --fix による275件の自動修正（Python 3.12モダン構文化、importソート等）
- per-file-ignores で upstream ML/NLP コードを適切に除外

### 変更 (Changed)
- `app.py`: Gradio UIを完全廃止し、FastAPI + HTMX のみの構成に変更
- 全UI画面（音声合成・モーフィング・ベクトル演算）をHTMX + Jinja2テンプレートで統一
- ナビゲーションバーをタブ形式に統一し、全ページ間で切り替え可能に

### 削除 (Removed)
- `gradio_tabs/` ディレクトリ（Gradio UIコード）を削除
- `gradio` パッケージをwebui依存から削除
- `tts_model.py` からGradio連携メソッド (`get_model_for_gradio()` 等) を削除
- `constants.py` から `GRADIO_THEME` を削除
- 学習・前処理・データセット作成関連のファイル・設定を削除（推論専用ブランチ化）

### 依存関係 (Dependencies)
- `jinja2` を webui 依存グループに追加 (テンプレートエンジン)
- `ruff` を style 依存グループに追加（`black[jupyter]` + `isort` を置き換え）
- `gradio` を webui 依存グループから削除

### 新規ファイル一覧

| ファイル | 説明 |
|----------|------|
| `style_bert_vits2/style_ops.py` | スタイルベクトル演算コアモジュール |
| `style_bert_vits2/expression_parser.py` | カスタム式パーサー |
| `inference_app.py` | 音声合成UI APIルーター |
| `morphing_app.py` | モーフィングUI APIルーター |
| `vector_app.py` | ベクトル演算UI APIルーター |
| `templates/inference.html` | 音声合成UIテンプレート |
| `templates/morphing.html` | モーフィングUIテンプレート |
| `templates/vector_arithmetic.html` | ベクトル演算UIテンプレート |
| `templates/partials/*.html` | HTMX部分更新フラグメント |
| `static/css/morphing.css` | ダークテーマスタイルシート |
| `tests/test_style_ops.py` | style_ops ユニットテスト |
| `tests/test_morphing_api.py` | モーフィングAPIテスト |
| `tests/test_expression_parser.py` | 式パーサーテスト |
| `tests/test_vector_app.py` | ベクトル演算テスト |
| `tests/test_edge_cases.py` | エッジケーステスト |
| `tests/test_integration.py` | 統合テスト |

過去のリリース履歴は [GitHub Releases](https://github.com/litagin02/Style-Bert-VITS2/releases/) を参照してください。
