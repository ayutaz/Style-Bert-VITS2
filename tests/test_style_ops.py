import json

import numpy as np
import pytest
from pathlib import Path

from style_bert_vits2.style_ops import (
    lerp,
    slerp,
    vector_add,
    vector_sub,
    vector_mean,
    vector_scale,
    vector_diff_transfer,
    clip_norm,
    compute_norm_ratio,
    validate_style_vector,
    load_style_vectors,
    save_style_vectors,
)


DIM = 256


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def v0(rng):
    return rng.standard_normal(DIM).astype(np.float32)


@pytest.fixture
def v1(rng):
    # 別のベクトルを生成するために2回呼ぶ (rngは状態を持つ)
    _ = rng.standard_normal(DIM)
    return rng.standard_normal(DIM).astype(np.float32)


# ============================================================
# Interpolation Tests
# ============================================================


class TestLerp:
    def test_t0_returns_v0(self, v0, v1):
        result = lerp(0.0, v0, v1)
        np.testing.assert_allclose(result, v0)

    def test_t1_returns_v1(self, v0, v1):
        result = lerp(1.0, v0, v1)
        np.testing.assert_allclose(result, v1)

    def test_t05_returns_midpoint(self, v0, v1):
        result = lerp(0.5, v0, v1)
        expected = (v0 + v1) / 2.0
        np.testing.assert_allclose(result, expected)

    def test_intermediate_value(self, v0, v1):
        result = lerp(0.3, v0, v1)
        expected = 0.7 * v0 + 0.3 * v1
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestSlerp:
    def test_t0_returns_v0(self, v0, v1):
        result = slerp(0.0, v0, v1)
        np.testing.assert_allclose(result, v0, rtol=1e-5)

    def test_t1_returns_v1(self, v0, v1):
        result = slerp(1.0, v0, v1)
        np.testing.assert_allclose(result, v1, rtol=1e-5)

    def test_t05_norm_is_interpolated(self, v0, v1):
        result = slerp(0.5, v0, v1)
        norm0 = np.linalg.norm(v0)
        norm1 = np.linalg.norm(v1)
        expected_norm = (norm0 + norm1) / 2.0
        np.testing.assert_allclose(
            np.linalg.norm(result), expected_norm, rtol=1e-5
        )

    def test_nearly_parallel_falls_back_to_lerp(self):
        """ほぼ同方向のベクトルではLERPにフォールバックする。"""
        v0 = np.ones(DIM, dtype=np.float32)
        v1 = v0 * 1.001  # ほぼ同方向
        result = slerp(0.5, v0, v1)
        expected = lerp(0.5, v0, v1)
        np.testing.assert_allclose(result, expected)

    def test_zero_vector_falls_back_to_lerp(self):
        """ゼロベクトルが含まれる場合はLERPにフォールバック。"""
        v0 = np.zeros(DIM, dtype=np.float32)
        v1 = np.ones(DIM, dtype=np.float32)
        result = slerp(0.5, v0, v1)
        expected = lerp(0.5, v0, v1)
        np.testing.assert_allclose(result, expected)

    def test_orthogonal_vectors(self):
        """直交ベクトルのSLERP。"""
        v0 = np.zeros(DIM, dtype=np.float64)
        v1 = np.zeros(DIM, dtype=np.float64)
        v0[0] = 1.0
        v1[1] = 1.0
        result = slerp(0.5, v0, v1)
        # 直交の場合、t=0.5で結果は45度方向
        expected_norm = 1.0  # (1+1)/2
        np.testing.assert_allclose(
            np.linalg.norm(result), expected_norm, rtol=1e-10
        )


# ============================================================
# Vector Arithmetic Tests
# ============================================================


class TestVectorArithmetic:
    def test_vector_add_basic(self, v0, v1):
        result = vector_add(v0, v1)
        np.testing.assert_allclose(result, v0 + v1)

    def test_vector_add_scaled(self, v0, v1):
        result = vector_add(v0, v1, scale=0.5)
        np.testing.assert_allclose(result, v0 + 0.5 * v1)

    def test_vector_add_zero_scale(self, v0, v1):
        result = vector_add(v0, v1, scale=0.0)
        np.testing.assert_allclose(result, v0)

    def test_vector_sub_basic(self, v0, v1):
        result = vector_sub(v0, v1)
        np.testing.assert_allclose(result, v0 - v1)

    def test_vector_sub_scaled(self, v0, v1):
        result = vector_sub(v0, v1, scale=0.5)
        np.testing.assert_allclose(result, v0 - 0.5 * v1)

    def test_vector_mean_two_vectors(self, v0, v1):
        result = vector_mean(v0, v1)
        expected = (v0 + v1) / 2.0
        np.testing.assert_allclose(result, expected)

    def test_vector_mean_three_vectors(self, rng):
        a = rng.standard_normal(DIM).astype(np.float32)
        b = rng.standard_normal(DIM).astype(np.float32)
        c = rng.standard_normal(DIM).astype(np.float32)
        result = vector_mean(a, b, c)
        expected = (a + b + c) / 3.0
        np.testing.assert_allclose(result, expected)

    def test_vector_mean_no_vectors_raises(self):
        with pytest.raises(ValueError):
            vector_mean()

    def test_vector_scale_basic(self, v0):
        result = vector_scale(v0, 2.0)
        np.testing.assert_allclose(result, v0 * 2.0)

    def test_vector_scale_zero(self, v0):
        result = vector_scale(v0, 0.0)
        np.testing.assert_allclose(result, np.zeros(DIM))

    def test_vector_scale_negative(self, v0):
        result = vector_scale(v0, -1.0)
        np.testing.assert_allclose(result, -v0)

    def test_vector_diff_transfer(self, rng):
        base = rng.standard_normal(DIM).astype(np.float32)
        source = rng.standard_normal(DIM).astype(np.float32)
        target = rng.standard_normal(DIM).astype(np.float32)
        result = vector_diff_transfer(base, source, target)
        expected = base + (target - source)
        np.testing.assert_allclose(result, expected)

    def test_vector_diff_transfer_scaled(self, rng):
        base = rng.standard_normal(DIM).astype(np.float32)
        source = rng.standard_normal(DIM).astype(np.float32)
        target = rng.standard_normal(DIM).astype(np.float32)
        result = vector_diff_transfer(base, source, target, scale=0.5)
        expected = base + 0.5 * (target - source)
        np.testing.assert_allclose(result, expected)

    def test_vector_diff_transfer_same_source_target(self, rng):
        """source == target の場合、結果はbaseと同じ。"""
        base = rng.standard_normal(DIM).astype(np.float32)
        v = rng.standard_normal(DIM).astype(np.float32)
        result = vector_diff_transfer(base, v, v)
        np.testing.assert_allclose(result, base)


# ============================================================
# Safety Tests
# ============================================================


class TestSafety:
    def test_clip_norm_clips_when_over_limit(self, rng):
        ref_vectors = rng.standard_normal((5, DIM)).astype(np.float32)
        mean_norm = np.mean(np.linalg.norm(ref_vectors, axis=1))
        max_factor = 2.0
        max_norm = mean_norm * max_factor

        # 上限を超えるベクトルを作成
        big_vec = rng.standard_normal(DIM).astype(np.float32)
        big_vec = big_vec / np.linalg.norm(big_vec) * (max_norm * 3.0)

        result = clip_norm(big_vec, ref_vectors, max_factor=max_factor)
        result_norm = np.linalg.norm(result)
        np.testing.assert_allclose(result_norm, max_norm, rtol=1e-5)

    def test_clip_norm_passes_when_under_limit(self, rng):
        ref_vectors = rng.standard_normal((5, DIM)).astype(np.float32)
        mean_norm = np.mean(np.linalg.norm(ref_vectors, axis=1))

        # 上限以下のベクトル
        small_vec = rng.standard_normal(DIM).astype(np.float32)
        small_vec = small_vec / np.linalg.norm(small_vec) * (mean_norm * 0.5)

        result = clip_norm(small_vec, ref_vectors, max_factor=2.0)
        np.testing.assert_allclose(result, small_vec)

    def test_clip_norm_direction_preserved(self, rng):
        """クリッピング後もベクトルの方向は保持される。"""
        ref_vectors = rng.standard_normal((5, DIM)).astype(np.float32)
        mean_norm = np.mean(np.linalg.norm(ref_vectors, axis=1))
        max_norm = mean_norm * 2.0

        big_vec = rng.standard_normal(DIM).astype(np.float32)
        big_vec = big_vec / np.linalg.norm(big_vec) * (max_norm * 5.0)

        result = clip_norm(big_vec, ref_vectors, max_factor=2.0)

        # 方向が同じか確認 (コサイン類似度が1に近い)
        cosine = np.dot(result, big_vec) / (
            np.linalg.norm(result) * np.linalg.norm(big_vec)
        )
        np.testing.assert_allclose(cosine, 1.0, rtol=1e-5)

    def test_compute_norm_ratio(self, rng):
        ref_vectors = rng.standard_normal((5, DIM)).astype(np.float32)
        mean_norm = np.mean(np.linalg.norm(ref_vectors, axis=1))

        vec = rng.standard_normal(DIM).astype(np.float32)
        vec_norm = np.linalg.norm(vec)

        ratio = compute_norm_ratio(vec, ref_vectors)
        expected = vec_norm / mean_norm
        np.testing.assert_allclose(ratio, expected, rtol=1e-5)

    def test_compute_norm_ratio_zero_reference(self):
        """参照ベクトル群がゼロの場合は0.0を返す。"""
        ref_vectors = np.zeros((3, DIM), dtype=np.float32)
        vec = np.ones(DIM, dtype=np.float32)
        ratio = compute_norm_ratio(vec, ref_vectors)
        assert ratio == 0.0

    def test_validate_style_vector_nan(self):
        vec = np.ones(DIM, dtype=np.float32)
        vec[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_style_vector(vec)

    def test_validate_style_vector_inf(self):
        vec = np.ones(DIM, dtype=np.float32)
        vec[0] = np.inf
        with pytest.raises(ValueError, match="Inf"):
            validate_style_vector(vec)

    def test_validate_style_vector_zero(self):
        vec = np.zeros(DIM, dtype=np.float32)
        with pytest.raises(ValueError, match="ゼロベクトル"):
            validate_style_vector(vec)

    def test_validate_style_vector_valid(self, v0):
        # 正常なベクトルではエラーが出ないことを確認
        validate_style_vector(v0)


# ============================================================
# Utility Tests (I/O)
# ============================================================


class TestUtilities:
    def _create_test_model(self, root_dir: Path, model_name: str, rng):
        """テスト用のモデルディレクトリを作成するヘルパー。"""
        model_dir = root_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        style2id = {"Neutral": 0, "Happy": 1, "Sad": 2}
        num_styles = len(style2id)
        vectors = rng.standard_normal((num_styles, DIM)).astype(np.float32)

        np.save(model_dir / "style_vectors.npy", vectors)

        config = {
            "data": {
                "style2id": style2id,
                "num_styles": num_styles,
                "training_files": "dummy",
            },
            "model": {"hidden_channels": 192},
        }
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return vectors, style2id, config

    def test_load_style_vectors(self, tmp_path, rng):
        model_name = "test_model"
        vectors, style2id, _ = self._create_test_model(
            tmp_path, model_name, rng
        )

        loaded_vectors, loaded_style2id = load_style_vectors(
            model_name, tmp_path
        )

        np.testing.assert_allclose(loaded_vectors, vectors)
        assert loaded_style2id == style2id

    def test_save_style_vectors(self, tmp_path, rng):
        model_name = "test_model"
        self._create_test_model(tmp_path, model_name, rng)

        # 新しいベクトルとstyle2idで保存
        new_style2id = {"Neutral": 0, "Happy": 1, "Angry": 2, "Whisper": 3}
        new_vectors = rng.standard_normal(
            (len(new_style2id), DIM)
        ).astype(np.float32)

        save_style_vectors(new_vectors, new_style2id, model_name, tmp_path)

        # ファイルを直接読み込んで検証
        saved_vectors = np.load(
            tmp_path / model_name / "style_vectors.npy"
        )
        np.testing.assert_allclose(saved_vectors, new_vectors)

        with open(
            tmp_path / model_name / "config.json", encoding="utf-8"
        ) as f:
            config = json.load(f)
        assert config["data"]["style2id"] == new_style2id
        assert config["data"]["num_styles"] == len(new_style2id)

    def test_save_preserves_other_config_fields(self, tmp_path, rng):
        """save時に既存のconfig.jsonの他のフィールドが保持される。"""
        model_name = "test_model"
        _, _, original_config = self._create_test_model(
            tmp_path, model_name, rng
        )

        new_style2id = {"Neutral": 0}
        new_vectors = rng.standard_normal((1, DIM)).astype(np.float32)
        save_style_vectors(new_vectors, new_style2id, model_name, tmp_path)

        with open(
            tmp_path / model_name / "config.json", encoding="utf-8"
        ) as f:
            config = json.load(f)

        # 他のフィールドが保持されていることを確認
        assert config["model"]["hidden_channels"] == 192
        assert config["data"]["training_files"] == "dummy"

    def test_round_trip(self, tmp_path, rng):
        """save -> load のラウンドトリップでデータが一致する。"""
        model_name = "test_model"
        self._create_test_model(tmp_path, model_name, rng)

        style2id = {"Neutral": 0, "Happy": 1, "Sad": 2, "Angry": 3}
        vectors = rng.standard_normal(
            (len(style2id), DIM)
        ).astype(np.float32)

        save_style_vectors(vectors, style2id, model_name, tmp_path)
        loaded_vectors, loaded_style2id = load_style_vectors(
            model_name, tmp_path
        )

        np.testing.assert_allclose(loaded_vectors, vectors)
        assert loaded_style2id == style2id
