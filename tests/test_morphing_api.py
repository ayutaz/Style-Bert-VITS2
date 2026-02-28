"""morphing_app.py の API テスト。"""

import numpy as np
import pytest

from style_bert_vits2.style_ops import (
    compute_norm_ratio,
    lerp,
    load_style_vectors,
    slerp,
    validate_style_vector,
)


class TestMorphingInterpolation:
    """補間ロジックのテスト（モデル不要）。"""

    def _make_vectors(self):
        """テスト用ダミーベクトルを生成。"""
        rng = np.random.default_rng(42)
        vec_a = rng.standard_normal(256).astype(np.float64)
        vec_b = rng.standard_normal(256).astype(np.float64)
        return vec_a, vec_b

    def test_slerp_endpoints(self):
        """SLERP t=0 で vec_a, t=1 で vec_b を返す。"""
        vec_a, vec_b = self._make_vectors()
        result_0 = slerp(0.0, vec_a, vec_b)
        result_1 = slerp(1.0, vec_a, vec_b)
        np.testing.assert_allclose(result_0, vec_a, atol=1e-6)
        np.testing.assert_allclose(result_1, vec_b, atol=1e-6)

    def test_lerp_endpoints(self):
        """LERP t=0 で vec_a, t=1 で vec_b を返す。"""
        vec_a, vec_b = self._make_vectors()
        result_0 = lerp(0.0, vec_a, vec_b)
        result_1 = lerp(1.0, vec_a, vec_b)
        np.testing.assert_allclose(result_0, vec_a, atol=1e-6)
        np.testing.assert_allclose(result_1, vec_b, atol=1e-6)

    def test_slerp_midpoint_valid(self):
        """SLERP t=0.5 の結果が有効なベクトル。"""
        vec_a, vec_b = self._make_vectors()
        result = slerp(0.5, vec_a, vec_b)
        validate_style_vector(result)

    def test_lerp_midpoint_valid(self):
        """LERP t=0.5 の結果が有効なベクトル。"""
        vec_a, vec_b = self._make_vectors()
        result = lerp(0.5, vec_a, vec_b)
        validate_style_vector(result)

    def test_norm_ratio_close_to_one(self):
        """同一ベクトル間の補間ノルム比は 1.0 に近い。"""
        vec_a, _ = self._make_vectors()
        vectors = np.stack([vec_a, vec_a * 1.01])
        result = slerp(0.5, vec_a, vec_a * 1.01)
        ratio = compute_norm_ratio(result, vectors)
        assert 0.9 < ratio < 1.1

    def test_interpolation_produces_intermediate_norm(self):
        """補間結果のノルムは両端の間にある。"""
        vec_a, vec_b = self._make_vectors()
        result = slerp(0.5, vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        norm_result = np.linalg.norm(result)
        min_norm = min(norm_a, norm_b) * 0.8
        max_norm = max(norm_a, norm_b) * 1.2
        assert min_norm <= norm_result <= max_norm


class TestMorphingE2E:
    """E2Eテスト（モデルが必要）。"""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        """テスト用モデルのセットアップ。モデルがなければスキップ。"""
        from pathlib import Path

        from style_bert_vits2.constants import BASE_DIR
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

    def test_load_style_vectors(self):
        """style_ops.load_style_vectors でベクトルを正しく読み込める。"""
        vectors, style2id = load_style_vectors(
            self.model_info.name, self.model_holder.root_dir
        )
        assert vectors.ndim == 2
        assert vectors.shape[1] == 256
        assert len(style2id) == vectors.shape[0]

    def test_morphing_synthesize(self):
        """補間ベクトルを使って音声合成ができる。"""
        from style_bert_vits2.constants import Languages

        vectors, style2id = load_style_vectors(
            self.model_info.name, self.model_holder.root_dir
        )
        styles = list(style2id.keys())
        vec_a = vectors[style2id[styles[0]]]
        if len(styles) >= 2:
            vec_b = vectors[style2id[styles[1]]]
        else:
            vec_b = vec_a * 1.05

        interpolated = slerp(0.5, vec_a, vec_b)
        validate_style_vector(interpolated)

        model = self.model_holder.get_model(self.model_info.name, self.model_path)
        sr, audio = model.infer(
            text="モーフィングテストです。",
            language=Languages.JP,
            speaker_id=0,
            style_vector_override=interpolated,
        )
        assert sr > 0
        assert len(audio) > 0
