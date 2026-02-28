from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ============================================================
# 1.1 Interpolation Functions
# ============================================================


def lerp(t: float, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """線形補間 (Linear Interpolation)。

    Args:
        t: 補間係数。0でv0、1でv1を返す。
        v0: 始点ベクトル。shape (256,)。
        v1: 終点ベクトル。shape (256,)。

    Returns:
        補間されたベクトル。shape (256,)。
    """
    return (1.0 - t) * v0 + t * v1


def slerp(
    t: float,
    v0: np.ndarray,
    v1: np.ndarray,
    dot_threshold: float = 0.9995,
) -> np.ndarray:
    """球面線形補間 (SLERP)。

    wespeaker埋め込みは非正規化のため、内部で正規化して補間後にノルムも
    線形補間する。ほぼ同方向 (cos > dot_threshold) の場合はLERPにフォールバック。

    Args:
        t: 補間係数。0でv0、1でv1を返す。
        v0: 始点ベクトル。shape (256,)。
        v1: 終点ベクトル。shape (256,)。
        dot_threshold: LERPにフォールバックするコサイン類似度の閾値。

    Returns:
        補間されたベクトル。shape (256,)。
    """
    norm0 = np.linalg.norm(v0)
    norm1 = np.linalg.norm(v1)

    # ゼロベクトルの場合はLERPにフォールバック
    if norm0 == 0.0 or norm1 == 0.0:
        return lerp(t, v0, v1)

    u0 = v0 / norm0
    u1 = v1 / norm1

    dot = np.clip(np.dot(u0, u1), -1.0, 1.0)

    # ほぼ同方向の場合はLERPにフォールバック
    if abs(dot) > dot_threshold:
        return lerp(t, v0, v1)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s0 = np.sin((1.0 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta

    unit_interp = s0 * u0 + s1 * u1

    # ノルムを線形補間
    norm_interp = (1.0 - t) * norm0 + t * norm1

    return unit_interp * norm_interp


# ============================================================
# 1.2 Vector Arithmetic Functions
# ============================================================


def vector_add(v0: np.ndarray, v1: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """スケール付き加算。

    Args:
        v0: ベースベクトル。shape (256,)。
        v1: 加算するベクトル。shape (256,)。
        scale: v1に適用するスケール係数。

    Returns:
        v0 + scale * v1。shape (256,)。
    """
    return v0 + scale * v1


def vector_sub(v0: np.ndarray, v1: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """スケール付き減算。

    Args:
        v0: ベースベクトル。shape (256,)。
        v1: 減算するベクトル。shape (256,)。
        scale: v1に適用するスケール係数。

    Returns:
        v0 - scale * v1。shape (256,)。
    """
    return v0 - scale * v1


def vector_mean(*vectors: np.ndarray) -> np.ndarray:
    """任意個数のベクトルの平均。

    Args:
        *vectors: 平均を取るベクトル群。各ベクトルのshapeは (256,)。

    Returns:
        平均ベクトル。shape (256,)。

    Raises:
        ValueError: ベクトルが1つも渡されなかった場合。
    """
    if len(vectors) == 0:
        raise ValueError("少なくとも1つのベクトルが必要です。")
    return np.mean(np.stack(vectors), axis=0)


def vector_scale(v: np.ndarray, scale: float) -> np.ndarray:
    """スカラー倍。

    Args:
        v: 入力ベクトル。shape (256,)。
        scale: スケール係数。

    Returns:
        v * scale。shape (256,)。
    """
    return v * scale


def vector_diff_transfer(
    base: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """差分転写。

    targetとsourceの差分をbaseに適用する。
    例: base + scale * (target - source)

    Args:
        base: ベースベクトル。shape (256,)。
        source: 差分の起点ベクトル。shape (256,)。
        target: 差分の終点ベクトル。shape (256,)。
        scale: 差分に適用するスケール係数。

    Returns:
        base + scale * (target - source)。shape (256,)。
    """
    return base + scale * (target - source)


# ============================================================
# 1.3 Safety Functions
# ============================================================


def clip_norm(
    vec: np.ndarray,
    reference_vectors: np.ndarray,
    max_factor: float = 2.0,
) -> np.ndarray:
    """ノルムクリッピング。

    参照ベクトル群の平均ノルムの max_factor 倍を上限としてクリッピングする。

    Args:
        vec: クリッピング対象のベクトル。shape (256,)。
        reference_vectors: 参照ベクトル群。shape (num_vectors, 256)。
        max_factor: 平均ノルムに対する上限倍率。

    Returns:
        クリッピングされたベクトル。shape (256,)。
    """
    norms = np.linalg.norm(reference_vectors, axis=1)
    mean_norm = np.mean(norms)
    max_norm = mean_norm * max_factor

    vec_norm = np.linalg.norm(vec)
    if vec_norm > max_norm and vec_norm > 0.0:
        vec = vec * (max_norm / vec_norm)

    return vec


def compute_norm_ratio(vec: np.ndarray, reference_vectors: np.ndarray) -> float:
    """参照に対するノルム比を返す (UI表示用)。

    Args:
        vec: 対象ベクトル。shape (256,)。
        reference_vectors: 参照ベクトル群。shape (num_vectors, 256)。

    Returns:
        norm(vec) / mean_norm(reference_vectors)。
    """
    norms = np.linalg.norm(reference_vectors, axis=1)
    mean_norm = np.mean(norms)
    if mean_norm == 0.0:
        return 0.0
    return float(np.linalg.norm(vec) / mean_norm)


def validate_style_vector(vec: np.ndarray) -> None:
    """NaN/Inf/ゼロベクトルチェック。

    問題がある場合はValueErrorを発生させる。

    Args:
        vec: チェック対象のベクトル。shape (256,)。

    Raises:
        ValueError: NaN、Inf、またはゼロベクトルが検出された場合。
    """
    if np.any(np.isnan(vec)):
        raise ValueError("スタイルベクトルにNaNが含まれています。")
    if np.any(np.isinf(vec)):
        raise ValueError("スタイルベクトルにInfが含まれています。")
    if np.linalg.norm(vec) == 0.0:
        raise ValueError("スタイルベクトルがゼロベクトルです。")


# ============================================================
# 1.4 Utility Functions
# ============================================================


def load_style_vectors(
    model_name: str,
    root_dir: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    """スタイルベクトルとstyle2idを読み込む。

    model_assets/{model_name}/style_vectors.npy と config.json から
    (vectors, style2id) を読み込んで返す。

    Args:
        model_name: モデル名。
        root_dir: model_assetsのルートディレクトリ。

    Returns:
        (vectors, style2id) のタプル。
        vectorsのshapeは (num_styles, 256)。
    """
    model_dir = root_dir / model_name
    vectors = np.load(model_dir / "style_vectors.npy")
    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    style2id: dict[str, int] = config["data"]["style2id"]
    return vectors, style2id


def save_style_vectors(
    vectors: np.ndarray,
    style2id: dict[str, int],
    model_name: str,
    root_dir: Path,
) -> None:
    """演算結果を保存する。

    style_vectors.npy を上書きし、config.json の data.style2id と
    data.num_styles も更新する。

    Args:
        vectors: 保存するスタイルベクトル。shape (num_styles, 256)。
        style2id: スタイル名からインデックスへのマッピング。
        model_name: モデル名。
        root_dir: model_assetsのルートディレクトリ。
    """
    model_dir = root_dir / model_name
    np.save(model_dir / "style_vectors.npy", vectors)

    config_path = model_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["data"]["style2id"] = style2id
    config["data"]["num_styles"] = len(style2id)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# ============================================================
# 1.5 Visualization Functions
# ============================================================


def pca_project_2d(vectors: np.ndarray) -> np.ndarray:
    """スタイルベクトルをPCAで2次元に射影する。

    Args:
        vectors: スタイルベクトル群。shape (n, 256)。n >= 2 である必要がある。

    Returns:
        2次元座標。shape (n, 2)。
    """
    if vectors.shape[0] < 2:
        # 1ベクトルの場合は原点に配置
        return np.zeros((vectors.shape[0], 2))

    centered = vectors - vectors.mean(axis=0)
    # SVD で主成分を抽出
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # 上位2成分に射影
    return centered @ Vt[:2].T


def prepare_plot_data(
    vectors: np.ndarray,
    style2id: dict[str, int],
    result_vector: np.ndarray | None = None,
    result_label: str = "結果",
) -> list[dict]:
    """可視化用の正規化済みプロットデータを準備する。

    PCA射影後、座標を5〜95の範囲に正規化する。

    Args:
        vectors: スタイルベクトル群。shape (num_styles, 256)。
        style2id: スタイル名→インデックスのマッピング。
        result_vector: 演算結果ベクトル（オプション）。shape (256,)。
        result_label: 結果ベクトルのラベル。

    Returns:
        プロットデータのリスト。各要素は
        {"x": float, "y": float, "label": str, "is_result": bool}。
    """
    labels = list(style2id.keys())
    is_result_flags = [False] * len(labels)

    all_vecs = vectors
    if result_vector is not None:
        all_vecs = np.vstack([vectors, result_vector.reshape(1, -1)])
        labels.append(result_label)
        is_result_flags.append(True)

    coords = pca_project_2d(all_vecs)

    # 座標を 5〜95 の範囲に正規化
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0  # ゼロ除算防止
    normalized = 5.0 + 90.0 * (coords - min_vals) / range_vals

    points = []
    for i, (x, y) in enumerate(normalized):
        points.append(
            {
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "label": labels[i],
                "is_result": is_result_flags[i],
            }
        )

    return points
