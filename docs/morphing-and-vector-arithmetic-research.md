# モーフィング・ベクトル演算機能 調査レポート

10エージェントによる並列調査の統合結果。実現可能性の評価と実装方針を報告する。

---

## 結論: 実現可能性

**両機能とも実現可能。** スタイルベクトル(256次元wespeaker埋め込み)はモデル推論の外側でNumPy配列として操作されるため、SLERP補間やベクトル演算を追加する際にモデル本体の変更は不要。既存コードベースにSLERP実装(`merge.py`)やスタイル演算パターンが既にあり、それらを活用できる。

---

## 1. スタイルベクトルシステムの現状

### データ構造
- **ファイル**: `model_assets/{model_name}/style_vectors.npy`
- **shape**: `(num_styles, 256)` — 各行が1つのスタイルの256次元wespeaker埋め込み
- **index 0**: 常に "Neutral"（全学習データの平均ベクトル）
- **dtype**: float32/float64（正規化されていない生のwespeaker出力）

### スタイル適用の数式
```
get_style_vector(style_id, weight):
    result = mean + (style_vec - mean) * weight
```
- `weight = 0.0` → Neutral (平均)
- `weight = 1.0` → そのスタイルそのまま
- `weight > 1.0` → スタイルの誇張（外挿）

### 対応する設定
- `config.json` の `data.style2id`: スタイル名→インデックスのマッピング
- `config.json` の `data.num_styles`: スタイル数（`style_vectors.npy`の行数と一致必須）
- `constants.py`: `DEFAULT_STYLE = "Neutral"`, `DEFAULT_STYLE_WEIGHT = 1.0`

---

## 2. 推論パイプラインでのスタイルベクトルの流れ

```
style (str, e.g., "Happy")
  → self.style2id[style] → style_id (int)
  → get_style_vector(style_id, weight) → NDArray shape=(256,)
  → infer.py: torch.from_numpy(style_vec).unsqueeze(0) → Tensor shape=(1, 256)
  → TextEncoder.forward():
      style_emb = self.style_proj(style_vec.unsqueeze(1))  # nn.Linear(256, hidden_channels)
      x = (emb + tone_emb + lang_emb + bert_emb + style_emb) * sqrt(hidden_channels)
  → エンコーダ → デュレーション予測 → フロー → ボコーダ → 音声波形
```

### 重要ポイント
- `style_proj` は `nn.Linear(256, hidden_channels)` — **線形変換**なので `proj(lerp(a,b,t)) = lerp(proj(a), proj(b), t)` が数学的に成立
- style_embは**ブロードキャスト**で全タイムステップに均一に加算される
- スタイルベクトルが直接参照されるのは**TextEncoderの入力埋め込みのみ**（以降は間接的影響）
- **Standard版とJP-Extra版でスタイル処理は同一**（違いはBERT入力数と判別器のみ）

---

## 3. 既存の関連実装

### merge.py にSLERP実装済み
```python
def slerp_tensors(t, v0, v1, dot_thres=0.998):
    dot = np.sum(v0c * v1c / (np.linalg.norm(v0c) * np.linalg.norm(v1c)))
    if abs(dot) > dot_thres:
        return lerp_tensors(t, v0, v1)  # フォールバック
    th0 = np.arccos(dot)
    return v0c * sin(th0 - th_t) / sin(th0) + v1c * sin(th_t) / sin(th0)
```
ただしこれはtorch.Tensor用。スタイルベクトル用にはNumPyネイティブ版を新規作成すべき。

### merge.py にスタイルベクトル演算パターン済み
- 通常マージ: `(1 - w) * A + w * B`
- 差分マージ: `A + w * (B - C)` — **ベクトル演算の基本操作そのもの**
- 加重和: `a * A + b * B + c * C`
- ヌルモデル加算: `A + w * B`

### TTSModel が NDArray を直接受け取れる設計済み
```python
# tts_model.py コンストラクタ
if isinstance(style_vec_path, np.ndarray):
    self.style_vectors = style_vec_path  # NDArray直接指定可能
```

---

## 4. ONNX推論との互換性

**完全に互換。** スタイルベクトルの演算はモデル推論の外側（NumPy空間）で行われるため、推論バックエンドに依存しない。

| 項目 | PyTorch | ONNX |
|------|---------|------|
| スタイルベクトル取得 | `get_style_vector()` | 同じ |
| テンソル変換 | `torch.from_numpy().unsqueeze(0)` | `np.expand_dims(axis=0)` |
| モデル内処理 | `style_proj` (nn.Linear) | ONNX演算子として変換済み |

唯一の注意点: ONNX推論では`NullModelParam`（推論時の動的重みマージ）は使用不可。

---

## 5. API拡張ポイント

### 現状の制約
| | server_fastapi.py | server_editor.py |
|---|---|---|
| スタイル名指定 | `style` (Query) | `style` (Body) |
| スタイル重み | `style_weight` (Query) | `styleWeight` (Body) |
| 音声参照 | `reference_audio_path` (Query) | なし |
| **直接ベクトル指定** | **不可** | **不可** |

### 推奨拡張
**最小変更**: `TTSModel.infer()` に `style_vector_override: Optional[NDArray] = None` パラメータを追加

```python
def infer(self, ..., style_vector_override: Optional[NDArray] = None, ...):
    if style_vector_override is not None:
        style_vector = style_vector_override
    elif reference_audio_path is not None:
        style_vector = self.get_style_vector_from_audio(...)
    else:
        style_vector = self.get_style_vector(style_id, style_weight)
```

既存APIとの後方互換性を完全に保持。

---

## 6. Gradio UI 拡張方針

### 現在のタブ構成 (6タブ)
1. 音声合成 — `create_inference_app(model_holder)`
2. データセット作成 — `create_dataset_app()`
3. 学習 — `create_train_app()`
4. スタイル作成 — `create_style_vectors_app()`
5. マージ — `create_merge_app(model_holder)`
6. ONNX変換 — `create_onnx_app(model_holder)`

### 推奨配置

**案A: 2つの独立タブとして追加**
```
1. 音声合成
2. データセット作成
3. 学習
4. スタイル作成
5. スタイルモーフィング (NEW)
6. ベクトル演算 (NEW)
7. マージ
8. ONNX変換
```

**案B: 1つの統合タブ（内部サブタブ）として追加 — 推奨**
```
1. 音声合成
2. データセット作成
3. 学習
4. スタイル作成
5. スタイル操作 (NEW: モーフィング + ベクトル演算のサブタブ)
6. マージ
7. ONNX変換
```
タブ数増加を抑制し、機能的に近いものをグループ化。

### 新タブの関数シグネチャ
```python
def create_style_operation_app(model_holder: TTSModelHolder) -> gr.Blocks:
```

### 再利用可能な既存UIパターン
- モデル選択: `model_name` → `model_path` の2段階Dropdown
- スタイル選択: `gr.Dropdown(choices=styles)`
- 重みスライダー: `gr.Slider(minimum=0, maximum=1, step=0.01)`
- 音声プレビュー: `gr.Audio(label="結果")`
- ベクトル可視化: `gr.Plot()` (matplotlibプロット)
- 動的UI: `@gr.render` デコレータ

---

## 7. SLERP実装方針

### NumPyネイティブ実装
```python
def slerp_style_vectors(
    t: float, v0: np.ndarray, v1: np.ndarray, dot_threshold: float = 0.9995
) -> np.ndarray:
    v0_norm = v0 / np.linalg.norm(v0)
    v1_norm = v1 / np.linalg.norm(v1)
    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    if abs(dot) > dot_threshold:
        return v0 * (1.0 - t) + v1 * t  # LERPフォールバック
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = np.sin(t * theta_0) / sin_theta_0
    return s0 * v0_norm + s1 * v1_norm
```

### デジェネレートケース
| ケース | 条件 | 対処 |
|--------|------|------|
| 同一方向 | `dot ≈ 1.0` | LERPフォールバック |
| 反対方向 | `dot ≈ -1.0` | 実用上ほぼ発生しない（同一話者のスタイル変種のため） |
| ゼロベクトル | `‖v‖ = 0` | 入力バリデーションで排除 |

### 正規化について
wespeaker埋め込みは**正規化されていない**。SLERPは単位球面上の補間のため、内部で正規化してから補間し、元のスケールを別途考慮する必要がある。

---

## 8. 品質・安定性リスクと安全策

### リスク

| レベル | weight範囲 | 症状 |
|--------|-----------|------|
| 軽度 | 2-5 | 声質の誇張、微妙な不自然さ |
| 中度 | 5-10 | 発音崩壊、不自然なイントネーション |
| 重度 | 10-20+ | ノイズ化、NaN発生、ゼロ長音声 |

ベクトル演算（加算・減算）では、結果ベクトルのノルムが学習時の分布から逸脱しやすい。`style_proj`が線形変換のため、入力ノルムがN倍なら出力もN倍になり、テキスト/BERT埋め込みを圧倒する。

### 必須の安全策: ノルムクリッピング

```python
def safe_style_vector(result_vec, reference_mean, max_norm_factor=2.0):
    max_norm = np.linalg.norm(reference_mean) * max_norm_factor
    current_norm = np.linalg.norm(result_vec)
    if current_norm > max_norm:
        result_vec = result_vec * (max_norm / current_norm)
    return result_vec
```

### 推奨する追加安全策
1. **UIにノルムメトリクス表示** — 演算結果のノルムを「安全/注意/危険」で色分け表示
2. **プレビュー再生** — 短いテキストで試し聴きできるボタン
3. **API側バリデーション** — `style_weight`に`ge=0.0, le=50.0`等の制約追加
4. **ノルムクリッピングをデフォルト有効** — 上級者向けに無効化オプション提供

---

## 9. モデル管理の制約

- `TTSModelHolder`は**同時に1モデルのみ**をロード（`current_model`は単一）
- スタイルベクトル（NumPy配列）は軽量なので**複数モデル分を同時にCPUメモリに保持可能**
- `np.load(root_dir / model_name / "style_vectors.npy")`で任意のモデルのスタイルベクトルを直接読み込める
- 異なるモデル間のスタイルベクトル演算は数学的に可能（同じwespeakerモデル由来の256次元空間）

---

## 10. 実装アーキテクチャ提案

### ファイル構成
```
style_bert_vits2/
  style_ops.py (NEW) — SLERP、ベクトル演算、ノルムクリッピングのユーティリティ
  tts_model.py — infer()にstyle_vector_overrideパラメータ追加

gradio_tabs/
  style_operations.py (NEW) — モーフィング + ベクトル演算のGradioタブ

app.py — 新タブの追加
server_fastapi.py — style_vectorパラメータ追加（オプショナル）
```

### style_ops.py の主要関数
```python
def slerp(t, v0, v1, dot_threshold=0.9995) -> np.ndarray
def lerp(t, v0, v1) -> np.ndarray
def vector_add(v0, v1, scale=1.0) -> np.ndarray
def vector_sub(v0, v1, scale=1.0) -> np.ndarray
def vector_mean(*vectors) -> np.ndarray
def vector_scale(v, scale) -> np.ndarray
def safe_clip_norm(vec, reference_mean, max_factor=2.0) -> np.ndarray
def load_style_vectors(model_name, root_dir) -> tuple[np.ndarray, dict[str, int]]
```

### データフロー
```
[UIでスタイル選択・演算指定]
  → style_ops.py で NumPy 演算
  → ノルムクリッピング適用
  → TTSModel.infer(style_vector_override=result_vec)
  → 既存の推論パイプライン（変更なし）
  → 音声出力
```

---

## 11. 既存merge.pyとの機能差分

| | 既存マージ (merge.py) | モーフィング | ベクトル演算 |
|---|---|---|---|
| 対象 | モデル重み + スタイルベクトル | スタイルベクトルのみ | スタイルベクトルのみ |
| 操作 | オフライン(永続保存) | リアルタイム(推論時) | リアルタイム(推論時) |
| 保存 | 新モデル作成 | オプショナル | オプショナル |
| 補間方法 | LERP/SLERP | SLERP(主) | 加減算/平均/スケーリング |
| UI複雑度 | 高(1500行超) | 中 | 中 |
