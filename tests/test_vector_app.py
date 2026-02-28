"""ベクトル演算 API のテスト。"""

import numpy as np
import pytest

from style_bert_vits2.style_ops import (
    clip_norm,
    compute_norm_ratio,
    validate_style_vector,
    vector_diff_transfer,
    vector_mean,
    vector_scale,
)


class TestVectorArithmeticOperations:
    """ベクトル演算ロジックのテスト（モデル不要）。"""

    def _make_vectors(self):
        """テスト用ダミーベクトルを生成。"""
        rng = np.random.default_rng(42)
        return rng, rng.standard_normal(256).astype(np.float64)

    # ---- diff transfer ----

    def test_diff_transfer_identity(self):
        """source == target のとき、結果 == base。"""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(256).astype(np.float64)
        source = rng.standard_normal(256).astype(np.float64)
        result = vector_diff_transfer(base, source, source, 1.0)
        np.testing.assert_allclose(result, base, atol=1e-10)

    def test_diff_transfer_scale(self):
        """scale=0 のとき結果 == base。"""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(256).astype(np.float64)
        source = rng.standard_normal(256).astype(np.float64)
        target = rng.standard_normal(256).astype(np.float64)
        result = vector_diff_transfer(base, source, target, 0.0)
        np.testing.assert_allclose(result, base, atol=1e-10)

    def test_diff_transfer_basic(self):
        """base + 1.0 * (target - source) を検証。"""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(256).astype(np.float64)
        source = rng.standard_normal(256).astype(np.float64)
        target = rng.standard_normal(256).astype(np.float64)
        expected = base + (target - source)
        result = vector_diff_transfer(base, source, target, 1.0)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    # ---- weighted mean ----

    def test_weighted_mean_equal_weights(self):
        """2つのベクトルの等重み平均 == vector_mean。"""
        rng = np.random.default_rng(42)
        vec_a = rng.standard_normal(256).astype(np.float64)
        vec_b = rng.standard_normal(256).astype(np.float64)
        expected = vector_mean(vec_a, vec_b)
        weighted = 0.5 * vec_a + 0.5 * vec_b
        np.testing.assert_allclose(weighted, expected, atol=1e-10)

    def test_weighted_mean_single(self):
        """1つのベクトルの平均 == そのベクトル自身。"""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(256).astype(np.float64)
        result = vector_mean(vec)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    # ---- scaling ----

    def test_scaling_identity(self):
        """scale=1.0 で元のベクトルと同じ。"""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(256).astype(np.float64)
        result = vector_scale(vec, 1.0)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_scaling_zero(self):
        """scale=0.0 でゼロベクトル。"""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(256).astype(np.float64)
        result = vector_scale(vec, 0.0)
        np.testing.assert_allclose(result, np.zeros(256), atol=1e-10)

    def test_scaling_double(self):
        """scale=2.0 で2倍。"""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(256).astype(np.float64)
        result = vector_scale(vec, 2.0)
        np.testing.assert_allclose(result, vec * 2.0, atol=1e-10)

    # ---- clip_norm ----

    def test_clip_norm_within_range(self):
        """ノルムが範囲内ならクリッピングされない。"""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((5, 256)).astype(np.float64)
        vec = ref[0]  # 参照に含まれるベクトル → 範囲内のはず
        clipped = clip_norm(vec, ref, max_factor=2.0)
        np.testing.assert_allclose(clipped, vec)

    def test_clip_norm_exceeds(self):
        """ノルムが範囲外ならクリッピングされる。"""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((5, 256)).astype(np.float64)
        vec = ref[0] * 100  # 極端に大きい
        clipped = clip_norm(vec, ref, max_factor=2.0)
        assert np.linalg.norm(clipped) < np.linalg.norm(vec)

    # ---- validate ----

    def test_validate_after_operations(self):
        """各演算結果が validate_style_vector をパスする。"""
        rng = np.random.default_rng(42)
        vec_a = rng.standard_normal(256).astype(np.float64)
        vec_b = rng.standard_normal(256).astype(np.float64)
        vec_c = rng.standard_normal(256).astype(np.float64)

        # diff_transfer
        result_dt = vector_diff_transfer(vec_a, vec_b, vec_c, 1.0)
        validate_style_vector(result_dt)

        # vector_mean
        result_mean = vector_mean(vec_a, vec_b)
        validate_style_vector(result_mean)

        # vector_scale (非ゼロ)
        result_scale = vector_scale(vec_a, 0.5)
        validate_style_vector(result_scale)


class TestVectorArithmeticE2E:
    """E2Eテスト（モデルが必要）。"""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        """テスト用モデルのセットアップ。モデルがなければスキップ。"""
        from pathlib import Path

        from style_bert_vits2.constants import BASE_DIR
        from style_bert_vits2.style_ops import load_style_vectors
        from style_bert_vits2.tts_model import TTSModelHolder

        model_root = BASE_DIR / "model_assets"
        if not model_root.exists():
            pytest.skip("model_assets ディレクトリが見つかりません")

        onnx_providers = [
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ]
        self.model_holder = TTSModelHolder(model_root, "cpu", onnx_providers)

        if len(self.model_holder.models_info) == 0:
            pytest.skip("音声合成モデルが見つかりません")

        # テスト用モデルを検索
        self.model_info = None
        for info in self.model_holder.models_info:
            if info.name in ("koharune-ami", "amitaro"):
                model_files = [
                    f
                    for f in info.files
                    if f.endswith(".safetensors") and not Path(f).name.startswith(".")
                ]
                if model_files:
                    self.model_info = info
                    self.model_path = model_files[0]
                    break

        if self.model_info is None:
            pytest.skip("テスト用モデル (koharune-ami/amitaro) が見つかりません")

        # スタイルベクトルを事前ロード
        self.vectors, self.style2id = load_style_vectors(
            self.model_info.name, self.model_holder.root_dir
        )

    def test_diff_transfer_synthesize(self):
        """差分転写結果で音声合成ができる。"""
        from style_bert_vits2.constants import Languages
        from style_bert_vits2.style_ops import (
            clip_norm,
            validate_style_vector,
            vector_diff_transfer,
        )

        styles = list(self.style2id.keys())
        base = self.vectors[self.style2id[styles[0]]]
        source = self.vectors[self.style2id[styles[-1]]]
        target = self.vectors[self.style2id[styles[min(1, len(styles) - 1)]]]

        result = vector_diff_transfer(base, source, target, 1.0)
        result = clip_norm(result, self.vectors)
        validate_style_vector(result)

        model = self.model_holder.get_model(self.model_info.name, self.model_path)
        sr, audio = model.infer(
            text="ベクトル演算テストです。",
            language=Languages.JP,
            speaker_id=0,
            style_vector_override=result,
        )
        assert sr > 0
        assert len(audio) > 0

    def test_scaling_synthesize(self):
        """スケーリング結果で音声合成ができる。"""
        from style_bert_vits2.constants import Languages
        from style_bert_vits2.style_ops import (
            clip_norm,
            validate_style_vector,
            vector_scale,
        )

        styles = list(self.style2id.keys())
        vec = self.vectors[self.style2id[styles[0]]]

        scaled = vector_scale(vec, 1.2)
        scaled = clip_norm(scaled, self.vectors)
        validate_style_vector(scaled)

        model = self.model_holder.get_model(self.model_info.name, self.model_path)
        sr, audio = model.infer(
            text="スケーリングテストです。",
            language=Languages.JP,
            speaker_id=0,
            style_vector_override=scaled,
        )
        assert sr > 0
        assert len(audio) > 0
