# モーフィング・ベクトル演算機能 ロードマップ

調査レポート (`morphing-and-vector-arithmetic-research.md`) に基づく実装計画。

---

## 進捗サマリー

| Phase | 内容 | 状態 | 進捗 |
|-------|------|------|------|
| **Phase 1** | コアライブラリ層 | **完了** | 100% |
| **Phase 2** | 推論パイプライン拡張 | **完了** | 100% |
| **Phase 3** | モーフィングUI | **完了** | 100% |
| **Phase 4** | ベクトル演算UI | **完了** | 100% |
| **Phase 5** | 統合・品質保証 | **完了** | 100% |

**全フェーズ完了。**

---

## フェーズ概要

```
Phase 1: コアライブラリ層        ← 完了
Phase 2: 推論パイプライン拡張    ← 完了
Phase 3: モーフィングUI          ← 完了
Phase 4: ベクトル演算UI          ← 完了
Phase 5: 統合・品質保証          ← 完了
```

```
 Phase 1 (完了) ─┬─→ Phase 3 (完了)
                  │                    ──→ Phase 5 (完了)
                  ├─→ Phase 4 (完了)
                  │
 Phase 2 (完了) ─┘
```

---

## Phase 1: コアライブラリ層 — 完了

**目的**: スタイルベクトル演算の基盤ユーティリティを実装する

**実装コミット**: `e9b18db` feat: Phase 1 スタイルベクトル演算モジュール (style_ops.py) を実装

### 実装済みファイル: `style_bert_vits2/style_ops.py` (290行)

#### 1.1 補間関数
| 関数 | 説明 |
|------|------|
| `lerp(t, v0, v1)` | 線形補間 (Linear Interpolation) |
| `slerp(t, v0, v1, dot_threshold=0.9995)` | 球面線形補間 (SLERP)。ほぼ同方向の場合LERPにフォールバック |

- 入力/出力: `np.ndarray` (shape: `(256,)`)
- wespeaker埋め込みは非正規化のため、SLERP内部で正規化して補間後にノルムも補間する方式を採用

#### 1.2 ベクトル演算関数
| 関数 | 説明 | 数式 |
|------|------|------|
| `vector_add(v0, v1, scale=1.0)` | スケール付き加算 | `v0 + scale * v1` |
| `vector_sub(v0, v1, scale=1.0)` | スケール付き減算 | `v0 - scale * v1` |
| `vector_mean(*vectors)` | 任意個数の平均 | `sum(vectors) / len(vectors)` |
| `vector_scale(v, scale)` | スカラー倍 | `v * scale` |
| `vector_diff_transfer(base, source, target, scale=1.0)` | 差分転写 | `base + scale * (target - source)` |

#### 1.3 安全策
| 関数 | 説明 |
|------|------|
| `clip_norm(vec, reference_vectors, max_factor=2.0)` | ノルムクリッピング。参照ベクトル群の平均ノルムの `max_factor` 倍を上限とする |
| `compute_norm_ratio(vec, reference_vectors)` | 参照に対するノルム比を返す（UI表示用） |
| `validate_style_vector(vec)` | NaN/Inf/ゼロベクトルチェック |

#### 1.4 ユーティリティ
| 関数 | 説明 |
|------|------|
| `load_style_vectors(model_name, root_dir)` | `style_vectors.npy` と `config.json` から `(vectors, style2id)` を読み込む |
| `save_style_vectors(vectors, style2id, model_name, root_dir)` | 演算結果を新しいスタイルとして保存 |

### 完了条件
- [x] 全関数の実装 (12関数)
- [x] 単体テスト (`tests/test_style_ops.py`, 43テストケース)
  - [x] SLERP: t=0でv0、t=1でv1、t=0.5で中間点
  - [x] SLERP: ほぼ同方向でLERPフォールバック
  - [x] ベクトル演算: 基本演算の正確性
  - [x] ノルムクリッピング: 上限超過時にクリップされる
  - [x] バリデーション: NaN/Inf/ゼロベクトルの検出
  - [x] I/O: load/save round-trip

---

## Phase 2: 推論パイプライン拡張 — 完了

**目的**: カスタムスタイルベクトルを推論パイプラインに注入可能にする

### 変更ファイル: `style_bert_vits2/tts_model.py`

#### 2.1 `TTSModel.infer()` にパラメータ追加
```python
def infer(
    self,
    ...,
    style_vector_override: Optional[NDArray[Any]] = None,  # NEW
    ...
):
```

- `style_vector_override` が指定された場合、`style` / `reference_audio_path` / `style_weight` を無視
- PyTorch推論・ONNX推論の両パスで動作
- 後方互換性を完全に保持（既存の呼び出しは影響なし）

#### 2.2 スタイルベクトル取得ロジックの変更

現在のコード (`tts_model.py:420-427`):
```python
# 変更前
if reference_audio_path is None:
    style_id = self.style2id[style]
    style_vector = self.get_style_vector(style_id, style_weight)
else:
    style_vector = self.get_style_vector_from_audio(...)
```

変更後:
```python
if style_vector_override is not None:
    style_vector = style_vector_override
elif reference_audio_path is not None:
    style_vector = self.get_style_vector_from_audio(...)
else:
    style_id = self.style2id[style]
    style_vector = self.get_style_vector(style_id, style_weight)
```

### 完了条件
- [x] `infer()` に `style_vector_override` パラメータ追加
- [x] PyTorch推論パスでの動作確認 (CPU / CUDA)
- [x] 既存テスト (`tests/test_main.py::test_synthesize_cpu`) が引き続きパス
- [x] 新規テスト追加 (4件): override基本動作、優先度確認、style_ops統合、CUDA

---

## Phase 3: モーフィングUI (HTMX) — 完了

**目的**: 2つのスタイル間のSLERP/LERP補間をHTMXベースのGUIで操作できるようにする

**設計変更**: 当初はGradioタブとして計画していたが、HTMXベースのスタンドアロンUIとして実装。
FastAPI + Jinja2テンプレート + HTMX による部分更新パターンを採用。

### 新規ファイル

| ファイル | 説明 |
|----------|------|
| `morphing_app.py` | FastAPI APIRouter — モーフィングUIのバックエンド |
| `templates/morphing.html` | メインページテンプレート |
| `templates/partials/style_options.html` | スタイル選択フラグメント |
| `templates/partials/audio_player.html` | 音声プレーヤーフラグメント |
| `templates/partials/norm_indicator.html` | ノルム安全インジケータ |
| `static/css/morphing.css` | ダークテーマスタイルシート |

### 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `app.py` | FastAPI + Gradio mount パターンに変更 |
| `pyproject.toml` | jinja2 依存追加 |

### APIエンドポイント

| Method | Path | 説明 |
|--------|------|------|
| GET | /morphing | メインページ |
| GET | /api/morphing/model-files | モデルファイル一覧 |
| POST | /api/morphing/load-model | モデルロード + スタイル取得 |
| POST | /api/morphing/compute-norm | ノルム比計算 |
| POST | /api/morphing/synthesize | 音声合成 |
| POST | /api/morphing/save-style | スタイル保存 |
| POST | /api/morphing/visualize | 2Dスタイルプロット (Phase 5) |

### 完了条件
- [x] モーフィングUI実装 (HTMX + Jinja2)
- [x] SLERP/LERP補間が正しく動作
- [x] 音声プレビュー再生
- [x] ノルム安全インジケータ表示
- [x] スタイル保存機能
- [x] 2Dプロット可視化 → Phase 5 で実装済み (`/api/morphing/visualize`)

---

## Phase 4: ベクトル演算UI (HTMX) — 完了

**目的**: スタイルベクトルの算術操作をGUIで行えるようにする

**設計変更**: 当初はGradioタブとして計画していたが、Phase 3と同様にHTMXベースのスタンドアロンUIとして実装。
FastAPI + Jinja2テンプレート + HTMX による部分更新パターンを採用。

### 新規ファイル

| ファイル | 説明 |
|----------|------|
| `vector_app.py` | FastAPI APIRouter — ベクトル演算UIのバックエンド |
| `style_bert_vits2/expression_parser.py` | カスタム式パーサー (eval不使用) |
| `templates/vector_arithmetic.html` | メインページテンプレート |
| `templates/partials/va_style_options.html` | モード別スタイル選択フラグメント |
| `tests/test_expression_parser.py` | 式パーサーユニットテスト |
| `tests/test_vector_app.py` | ベクトル演算APIテスト |

### 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `app.py` | vector_router 統合 + ナビゲーションリンク追加 |
| `templates/morphing.html` | ベクトル演算ページへのリンク追加 |
| `static/css/morphing.css` | ベクトル演算UI用のCSS追加 |

### APIエンドポイント

| Method | Path | 説明 |
|--------|------|------|
| GET | /vector-arithmetic | メインページ |
| GET | /api/vector/model-files | モデルファイル一覧 |
| POST | /api/vector/load-model | モデルロード + スタイル取得 |
| POST | /api/vector/compute-norm | ノルム比計算 |
| POST | /api/vector/synthesize | 音声合成 |
| POST | /api/vector/save-style | スタイル保存 |
| POST | /api/vector/visualize | 2Dスタイルプロット (Phase 5) |

### 演算モード

| モード | 数式 | 説明 |
|--------|------|------|
| 差分転写 | A + scale × (B − C) | ベースにターゲット-ソース差分を適用 |
| 加重平均 | w₁×A + w₂×B + w₃×C | 重み付き平均（正規化オプション付き） |
| スケーリング | scale × A | スタイルの強調・抑制 |
| カスタム式 | ユーザー定義 | 四則演算 + スタイル名の自由式 |

### 完了条件
- [x] 4つの演算モードUI実装
- [x] 各モードの演算ロジック
- [x] ノルムクリッピングのトグルと倍率設定
- [x] 音声プレビュー
- [x] スタイル保存機能
- [x] カスタム式パーサー（基本的な四則演算 + スタイル名参照）

---

## Phase 5: 統合・品質保証 — 完了

**目的**: 全フェーズの統合テストとドキュメント整備

### 5.1 app.py への統合

HTMXベースのUIとして実装済み。`app.py` で FastAPI APIRouter をマウント:

```python
from morphing_app import router as morphing_router
from vector_app import router as vector_router

fastapi_app.include_router(morphing_router)
fastapi_app.include_router(vector_router)
```

- `/morphing` — モーフィングUI
- `/vector-arithmetic` — ベクトル演算UI

### 5.2 テスト
- [x] `tests/test_style_ops.py` — ユニットテスト回帰確認
- [x] 既存テストの回帰確認 — `uv run pytest` がパス
- [x] ONNX推論での動作確認 — style_vector_override はONNX分岐前に適用されるため互換性あり
- [x] エッジケース — NaN/Inf、極端なスケール値、スタイルが1つしかないモデル (`tests/test_edge_cases.py`)
- [x] 統合テスト — モジュールインポート、API一貫性 (`tests/test_integration.py`)

### 5.3 コードスタイル
- [x] `uv run black --check .` がパス (全67ファイル、bert_models.py・tts_model.py の `# fmt: skip` 起因バグも修正済み)
- [x] `uv run isort --check-only --profile black .` がパス
- [x] docstring監査完了（日本語、Google style）

### 5.4 ドキュメント
- [x] `docs/CHANGELOG.md` にリリースノート追加
- [x] WebUIの使い方説明（HTMLの<details>要素によるガイドセクション）
- [x] 2Dスタイルベクトル可視化 — PCA射影SVGプロット

---

## 変更ファイル一覧

| ファイル | 変更種別 | Phase | 状態 |
|----------|----------|-------|------|
| `style_bert_vits2/style_ops.py` | 変更 | 1, 5 | **完了** |
| `tests/test_style_ops.py` | **新規** | 1 | **完了** |
| `style_bert_vits2/tts_model.py` | 変更 | 2, 5 | **完了** |
| `morphing_app.py` | **新規** | 3, 5 | **完了** |
| `templates/partials/*` | **新規** | 3, 4, 5 | **完了** |
| `static/css/morphing.css` | 変更 | 3, 4, 5 | **完了** |
| `app.py` | 変更 | 3, 4 | **完了** |
| `pyproject.toml` | 変更 | 3, 5 | **完了** |
| `tests/test_morphing_api.py` | **新規** | 3 | **完了** |
| `templates/morphing.html` | 変更 | 3, 4, 5 | **完了** |
| `vector_app.py` | **新規** | 4 | **完了** |
| `style_bert_vits2/expression_parser.py` | **新規** | 4 | **完了** |
| `templates/vector_arithmetic.html` | **新規** | 4 | **完了** |
| `templates/partials/va_style_options.html` | **新規** | 4 | **完了** |
| `tests/test_expression_parser.py` | **新規** | 4 | **完了** |
| `tests/test_vector_app.py` | **新規** | 4 | **完了** |
| `tests/test_edge_cases.py` | **新規** | 5 | **完了** |
| `tests/test_integration.py` | **新規** | 5 | **完了** |
| `docs/CHANGELOG.md` | **新規** | 5 | **完了** |
| `templates/partials/style_plot.html` | **新規** | 5 | **完了** |
| `style_bert_vits2/nlp/bert_models.py` | 変更 (black修正) | 5 | **完了** |

---

## リスク管理

| リスク | 影響 | 対策 |
|--------|------|------|
| ノルム逸脱による音質崩壊 | 高 | ノルムクリッピング(デフォルト有効) + UIインジケータ |
| 異なるモデル間のスタイルベクトル非互換 | 中 | 同一モデル内での演算に限定（UI制約） |
| Gradioバージョン互換性 | 低 | 既存UIパターンを踏襲 (`>=4.32`) |
| ONNX推論での未対応 | 低 | スタイルベクトル演算はモデル外部のため影響なし |
| 既存機能への回帰 | 中 | `style_vector_override=None`のデフォルト値で後方互換保持 |
