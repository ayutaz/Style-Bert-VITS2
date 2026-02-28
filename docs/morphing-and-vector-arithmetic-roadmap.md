# モーフィング・ベクトル演算機能 ロードマップ

調査レポート (`morphing-and-vector-arithmetic-research.md`) に基づく実装計画。

---

## フェーズ概要

```
Phase 1: コアライブラリ層        ← 他の全フェーズの基盤
Phase 2: 推論パイプライン拡張    ← Phase 1 に依存
Phase 3: モーフィングUI          ← Phase 1, 2 に依存
Phase 4: ベクトル演算UI          ← Phase 1, 2 に依存
Phase 5: 統合・品質保証          ← 全フェーズ完了後
```

```
 Phase 1 ─┬─→ Phase 3 (モーフィングUI)
           │
           ├─→ Phase 4 (ベクトル演算UI)    ──→ Phase 6 (統合・QA)
           │
           └─→ Phase 5 (API拡張)
           │
 Phase 2 ──┘
```

---

## Phase 1: コアライブラリ層

**目的**: スタイルベクトル演算の基盤ユーティリティを実装する

### 新規ファイル: `style_bert_vits2/style_ops.py`

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
- [ ] 全関数の実装
- [ ] 単体テスト (`tests/test_style_ops.py`)
  - SLERP: t=0でv0、t=1でv1、t=0.5で中間点
  - SLERP: ほぼ同方向でLERPフォールバック
  - ベクトル演算: 基本演算の正確性
  - ノルムクリッピング: 上限超過時にクリップされる
  - バリデーション: NaN/Inf/ゼロベクトルの検出

---

## Phase 2: 推論パイプライン拡張

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
```python
# 変更前
if reference_audio_path is None:
    style_id = self.style2id[style]
    style_vector = self.get_style_vector(style_id, style_weight)
else:
    style_vector = self.get_style_vector_from_audio(...)

# 変更後
if style_vector_override is not None:
    style_vector = style_vector_override
elif reference_audio_path is not None:
    style_vector = self.get_style_vector_from_audio(...)
else:
    style_id = self.style2id[style]
    style_vector = self.get_style_vector(style_id, style_weight)
```

### 完了条件
- [ ] `infer()` にパラメータ追加
- [ ] PyTorch推論パスでの動作確認
- [ ] ONNX推論パスでの動作確認
- [ ] 既存テスト (`tests/test_main.py`) が引き続きパスすること

---

## Phase 3: モーフィングUI (Gradio タブ)

**目的**: 2つのスタイル間のSLERP補間をGUIで操作できるようにする

### 新規ファイル: `gradio_tabs/style_operations.py`

#### 3.1 モーフィングサブタブのUI構成

```
┌─────────────────────────────────────────────────────┐
│ スタイル操作 > モーフィング                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  モデル選択: [Dropdown: model_name]                  │
│  モデルファイル: [Dropdown: model_path]               │
│                                                     │
│  ┌─ スタイルA ─────────┐  ┌─ スタイルB ─────────┐   │
│  │ [Dropdown: style_A] │  │ [Dropdown: style_B] │   │
│  └─────────────────────┘  └─────────────────────┘   │
│                                                     │
│  補間方法: ○ SLERP (推奨)  ○ LERP                   │
│                                                     │
│  補間率 (t): [====●===========] 0.30                 │
│              A (0.0)        B (1.0)                  │
│                                                     │
│  ノルム状態: ● 安全 (比率: 1.02)                     │
│                                                     │
│  テキスト: [こんにちは、今日はいい天気ですね。]        │
│                                                     │
│  [音声プレビュー]  [スタイルとして保存]               │
│                                                     │
│  結果: [▶ Audio Player]                              │
│                                                     │
│  ┌─ ベクトル可視化 ─────────────────────────────┐   │
│  │  [2D Plot: A, B, 補間結果の位置を表示]        │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

#### 3.2 主要機能
1. **モデル選択** — 既存パターン（model_name → model_path の2段階Dropdown）を再利用
2. **スタイルA/B選択** — config.jsonのstyle2idからDropdown生成
3. **補間方法** — SLERP / LERP のRadio選択
4. **補間率スライダー** — `t: 0.0 ~ 1.0`、step=0.01
5. **ノルム安全インジケータ** — 結果ベクトルのノルム比を色分け表示
6. **音声プレビュー** — 補間結果で即座にTTS実行
7. **保存機能** — 結果を新しいスタイルとして `style_vectors.npy` に追加
8. **2Dプロット** — スタイルA, B, Neutral, 補間結果をUMAP/PCA等で2D可視化

### 完了条件
- [ ] モーフィングサブタブUI実装
- [ ] SLERP/LERP補間が正しく動作
- [ ] 音声プレビュー再生
- [ ] ノルム安全インジケータ表示
- [ ] スタイル保存機能
- [ ] 2Dプロット可視化

---

## Phase 4: ベクトル演算UI (Gradio タブ)

**目的**: スタイルベクトルの算術操作をGUIで行えるようにする

#### 4.1 ベクトル演算サブタブのUI構成

```
┌─────────────────────────────────────────────────────┐
│ スタイル操作 > ベクトル演算                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  モデル選択: [Dropdown: model_name]                  │
│  モデルファイル: [Dropdown: model_path]               │
│                                                     │
│  演算モード:                                         │
│    ○ 差分転写  A + scale × (B - C)                   │
│    ○ 加重平均  w1×A + w2×B + w3×C                    │
│    ○ スケーリング  scale × A                          │
│    ○ カスタム式                                      │
│                                                     │
│  ── 差分転写モード ──                                │
│  ベース (A):    [Dropdown]                           │
│  ソース (C):    [Dropdown]   → 差分の「元」           │
│  ターゲット (B): [Dropdown]  → 差分の「先」           │
│  スケール:      [====●=====] 1.0                     │
│                                                     │
│  □ ノルムクリッピング有効 (推奨)                      │
│  クリッピング倍率: [====●=====] 2.0                   │
│                                                     │
│  ノルム状態: ▲ 注意 (比率: 1.85)                     │
│                                                     │
│  テキスト: [こんにちは、今日はいい天気ですね。]        │
│                                                     │
│  [音声プレビュー]  [スタイルとして保存]               │
│                                                     │
│  結果: [▶ Audio Player]                              │
│                                                     │
│  ┌─ ベクトル可視化 ─────────────────────────────┐   │
│  │  [2D Plot: 全スタイル + 演算結果を表示]        │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

#### 4.2 演算モード詳細

| モード | 数式 | ユースケース |
|--------|------|-------------|
| 差分転写 | `A + scale × (B - C)` | 「男性声Aに女性声Bの特徴を付与」 |
| 加重平均 | `w1×A + w2×B + w3×C` | 「3人の声の中間的な声」 |
| スケーリング | `scale × A` | 「スタイルの強調/抑制」 |
| カスタム式 | ユーザー定義 | 上級者向け自由演算 |

#### 4.3 カスタム式モード（上級者向け）
- テキストボックスで式を入力: `Happy - 0.3 * Neutral + 0.5 * Sad`
- スタイル名を変数として認識し、対応するベクトルで演算
- 安全なパーサー（evalは使わない）で処理

### 完了条件
- [ ] 4つの演算モードUI実装
- [ ] 各モードの演算ロジック
- [ ] ノルムクリッピングのトグルと倍率設定
- [ ] 音声プレビュー
- [ ] スタイル保存機能
- [ ] カスタム式パーサー（基本的な四則演算 + スタイル名参照）

---

## Phase 5: 統合・品質保証

**目的**: 全フェーズの統合テストとドキュメント整備

### 5.1 app.py への統合
```python
# app.py に追加
from gradio_tabs.style_operations import create_style_operation_app

with gr.Blocks(theme=GRADIO_THEME) as app:
    gr.Markdown(f"# Style-Bert-VITS2 WebUI (version {VERSION})")
    create_inference_app(model_holder=model_holder)
    create_style_operation_app(model_holder=model_holder)  # NEW
```

### 5.2 テスト
- [ ] `tests/test_style_ops.py` — ユニットテスト
- [ ] モーフィングUI手動テスト — 複数モデルで音声生成確認
- [ ] ベクトル演算UI手動テスト — 各演算モードの動作確認
- [ ] 既存テストの回帰確認 — `uv run pytest` がパス
- [ ] ONNX推論での動作確認
- [ ] エッジケース — NaN/Inf、極端なスケール値、スタイルが1つしかないモデル

### 5.3 コードスタイル
- [ ] `uv run black --check .` がパス
- [ ] docstring追加（日本語、Google style）

### 5.4 ドキュメント
- [ ] `docs/CHANGELOG.md` にリリースノート追加
- [ ] WebUIの使い方説明（Accordion内のMarkdown）

---

## 変更ファイル一覧

| ファイル | 変更種別 | Phase |
|----------|----------|-------|
| `style_bert_vits2/style_ops.py` | **新規** | 1 |
| `tests/test_style_ops.py` | **新規** | 1 |
| `style_bert_vits2/tts_model.py` | 変更 | 2 |
| `gradio_tabs/style_operations.py` | **新規** | 3, 4 |
| `app.py` | 変更 | 5 |
| `docs/CHANGELOG.md` | 変更 | 5 |

---

## リスク管理

| リスク | 影響 | 対策 |
|--------|------|------|
| ノルム逸脱による音質崩壊 | 高 | ノルムクリッピング(デフォルト有効) + UIインジケータ |
| 異なるモデル間のスタイルベクトル非互換 | 中 | 同一モデル内での演算に限定（UI制約） |
| Gradioバージョン互換性 | 低 | 既存UIパターンを踏襲 (`>=4.32`) |
| ONNX推論での未対応 | 低 | スタイルベクトル演算はモデル外部のため影響なし |
| 既存機能への回帰 | 中 | `style_vector_override=None`のデフォルト値で後方互換保持 |
