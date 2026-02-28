"""エッジケーステスト（Phase 5 品質保証）。"""

import numpy as np
import pytest

from style_bert_vits2.expression_parser import parse_expression
from style_bert_vits2.style_ops import (
    clip_norm,
    compute_norm_ratio,
    lerp,
    slerp,
    validate_style_vector,
    vector_diff_transfer,
    vector_mean,
    vector_scale,
)


DIM = 256


@pytest.fixture
def rng():
    return np.random.default_rng(99)


# ============================================================
# 補間のエッジケース
# ============================================================


class TestInterpolationEdgeCases:
    def test_slerp_identical_vectors(self):
        """同一ベクトルでslerp → 元ベクトルと同じ。"""
        v = np.random.default_rng(0).standard_normal(DIM).astype(np.float32)
        result = slerp(0.5, v, v)
        np.testing.assert_allclose(result, v, rtol=1e-5)

    def test_slerp_opposite_vectors(self):
        """正反対のベクトル (-v) でslerp。"""
        v = np.random.default_rng(1).standard_normal(DIM).astype(np.float32)
        neg_v = -v
        # 正反対のベクトル: cos = -1.0 なので dot_threshold 条件外で
        # SLERP が実行されるが、theta = pi で sin(pi) ≈ 0 のため数値的に不安定。
        # エラーなく結果が返ることを確認する。
        result = slerp(0.5, v, neg_v)
        assert result.shape == (DIM,)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_slerp_near_zero_vectors(self):
        """ノルムが極小のベクトルでslerp → LERPフォールバック。"""
        v0 = np.ones(DIM, dtype=np.float32) * 1e-40
        v1 = np.ones(DIM, dtype=np.float32) * 2e-40
        # ノルムが 0.0 として扱われるので LERP にフォールバック
        result = slerp(0.5, v0, v1)
        expected = lerp(0.5, v0, v1)
        np.testing.assert_allclose(result, expected)

    def test_lerp_out_of_range_t(self, rng):
        """t=-0.5, t=1.5 での外挿（エラーなく動作）。"""
        v0 = rng.standard_normal(DIM).astype(np.float32)
        v1 = rng.standard_normal(DIM).astype(np.float32)

        # t = -0.5: 外挿
        result_neg = lerp(-0.5, v0, v1)
        expected_neg = 1.5 * v0 + (-0.5) * v1
        np.testing.assert_allclose(result_neg, expected_neg, rtol=1e-5)

        # t = 1.5: 外挿
        result_over = lerp(1.5, v0, v1)
        expected_over = (-0.5) * v0 + 1.5 * v1
        np.testing.assert_allclose(result_over, expected_over, rtol=1e-5)

    def test_slerp_out_of_range_t(self, rng):
        """slerpでもt=-0.5, t=1.5の外挿がエラーなく動作する。"""
        v0 = rng.standard_normal(DIM).astype(np.float32)
        v1 = rng.standard_normal(DIM).astype(np.float32)

        result_neg = slerp(-0.5, v0, v1)
        assert result_neg.shape == (DIM,)
        assert not np.any(np.isnan(result_neg))

        result_over = slerp(1.5, v0, v1)
        assert result_over.shape == (DIM,)
        assert not np.any(np.isnan(result_over))


# ============================================================
# ベクトル演算のエッジケース
# ============================================================


class TestVectorArithmeticEdgeCases:
    def test_vector_scale_extreme_large(self, rng):
        """scale=1000 → validate_style_vector は通る（Infにならない）。"""
        v = rng.standard_normal(DIM).astype(np.float64)
        result = vector_scale(v, 1000.0)
        # Inf にならないことを確認
        assert not np.any(np.isinf(result))
        # validate_style_vector を通過する
        validate_style_vector(result)

    def test_vector_scale_negative(self, rng):
        """scale=-1.0 → 方向反転。"""
        v = rng.standard_normal(DIM).astype(np.float32)
        result = vector_scale(v, -1.0)
        np.testing.assert_allclose(result, -v)

    def test_vector_diff_transfer_all_same(self, rng):
        """base==source==target → 結果はbase。"""
        v = rng.standard_normal(DIM).astype(np.float32)
        result = vector_diff_transfer(v, v, v)
        np.testing.assert_allclose(result, v)

    def test_vector_mean_many(self, rng):
        """100個のベクトルの平均。"""
        vectors = [rng.standard_normal(DIM).astype(np.float32) for _ in range(100)]
        result = vector_mean(*vectors)
        expected = np.mean(np.stack(vectors), axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_vector_mean_weighted_zero_sum(self):
        """重みの合計が0のケースをvector_scaleとvector_meanの組み合わせでシミュレート。

        0*A + 0*B の平均 → ゼロベクトル。
        """
        a = np.ones(DIM, dtype=np.float32) * 3.0
        b = np.ones(DIM, dtype=np.float32) * 7.0
        scaled_a = vector_scale(a, 0.0)
        scaled_b = vector_scale(b, 0.0)
        result = vector_mean(scaled_a, scaled_b)
        np.testing.assert_allclose(result, np.zeros(DIM, dtype=np.float32))


# ============================================================
# 安全機構のエッジケース
# ============================================================


class TestSafetyEdgeCases:
    def test_clip_norm_zero_vector(self, rng):
        """ゼロベクトルをクリップ → そのまま返す。"""
        ref_vectors = rng.standard_normal((5, DIM)).astype(np.float32)
        zero_vec = np.zeros(DIM, dtype=np.float32)
        result = clip_norm(zero_vec, ref_vectors)
        np.testing.assert_allclose(result, zero_vec)

    def test_clip_norm_single_reference(self, rng):
        """参照ベクトルが1つだけでもclip_normが動作する。"""
        ref_vectors = rng.standard_normal((1, DIM)).astype(np.float32)
        mean_norm = np.linalg.norm(ref_vectors[0])
        max_norm = mean_norm * 2.0

        # 上限を超えるベクトルを作成
        big_vec = rng.standard_normal(DIM).astype(np.float32)
        big_vec = big_vec / np.linalg.norm(big_vec) * (max_norm * 3.0)

        result = clip_norm(big_vec, ref_vectors, max_factor=2.0)
        result_norm = np.linalg.norm(result)
        np.testing.assert_allclose(result_norm, max_norm, rtol=1e-5)

    def test_compute_norm_ratio_zero_reference_vectors(self):
        """全参照がゼロベクトル → 0.0を返す。"""
        ref_vectors = np.zeros((5, DIM), dtype=np.float32)
        vec = np.ones(DIM, dtype=np.float32)
        ratio = compute_norm_ratio(vec, ref_vectors)
        assert ratio == 0.0

    def test_validate_very_large_values(self):
        """極端に大きい値（1e30）→ パスする（Infではないので）。"""
        vec = np.ones(DIM, dtype=np.float64) * 1e30
        # ValueError が発生しないことを確認
        validate_style_vector(vec)

    def test_validate_very_small_values(self):
        """極端に小さい値（1e-30）→ パスする（ゼロではないので）。"""
        vec = np.ones(DIM, dtype=np.float64) * 1e-30
        # ノルムは sqrt(256) * 1e-30 ≈ 1.6e-29 > 0 なので通る
        validate_style_vector(vec)


# ============================================================
# 式パーサーのエッジケース
# ============================================================


class TestParserEdgeCases:
    def test_parser_style_name_with_number_prefix(self):
        """数字で始まるスタイル名はトークナイザが数値を先に消費するためValueError。"""
        available = ["123style", "Neutral"]
        with pytest.raises(ValueError):
            parse_expression("123style", available)

    def test_parser_style_name_non_ascii(self):
        """日本語スタイル名が正しくマッチする。"""
        available = ["喜び", "悲しみ", "Neutral"]
        result = parse_expression("喜び + 0.5 * 悲しみ", available)
        assert result == [(1.0, "喜び"), (0.5, "悲しみ")]

    def test_parser_consecutive_operators(self):
        """'Happy + + Sad' → ValueError。

        最初の + は二項演算子として消費され、次の + は単項符号として消費、
        その後 Sad がスタイルとしてパースされる。
        文法上 '+ +' は 'term = [sign] ...' で sign='+' が消費されるので
        パースが成功する場合がある。実際の挙動を検証する。
        """
        available = ["Happy", "Sad"]
        # + + は「二項演算子 + 」＋「term の単項符号 +」として解釈される
        result = parse_expression("Happy + + Sad", available)
        assert result == [(1.0, "Happy"), (1.0, "Sad")]

    def test_parser_trailing_operator(self):
        """'Happy +' → ValueError。"""
        available = ["Happy", "Sad"]
        with pytest.raises(ValueError):
            parse_expression("Happy +", available)

    def test_parser_only_number(self):
        """'0.5' → ValueError（スタイル名なし）。"""
        available = ["Happy", "Sad"]
        with pytest.raises(ValueError):
            parse_expression("0.5", available)

    def test_parser_very_long_expression(self):
        """20項の長い式がパースできる。"""
        styles = [f"Style{i}" for i in range(20)]
        expression = " + ".join(f"0.5 * {s}" for s in styles)
        result = parse_expression(expression, styles)
        assert len(result) == 20
        for coeff, name in result:
            assert coeff == pytest.approx(0.5)
            assert name in styles

    def test_parser_duplicate_style(self):
        """'Happy + Happy' → [(1.0, 'Happy'), (1.0, 'Happy')] (重複OK)。"""
        available = ["Happy", "Sad"]
        result = parse_expression("Happy + Happy", available)
        assert result == [(1.0, "Happy"), (1.0, "Happy")]

    def test_parser_negative_coefficient(self):
        """'-0.5 * Happy' → [(-0.5, 'Happy')]。"""
        available = ["Happy", "Sad"]
        result = parse_expression("-0.5 * Happy", available)
        assert result == [(-0.5, "Happy")]

    def test_parser_style_name_substring(self):
        """available_styles=['Happy', 'HappyDay'] で 'HappyDay' が正しくマッチ。

        長い名前から先にマッチされるため部分一致にならない。
        """
        available = ["Happy", "HappyDay"]
        result = parse_expression("HappyDay", available)
        assert result == [(1.0, "HappyDay")]

    def test_parser_style_name_substring_in_expression(self):
        """'HappyDay + Happy' で両方が正しくマッチする。"""
        available = ["Happy", "HappyDay"]
        result = parse_expression("HappyDay + Happy", available)
        assert result == [(1.0, "HappyDay"), (1.0, "Happy")]


# ============================================================
# 単一スタイルモデルシミュレーション
# ============================================================


class TestSingleStyleModel:
    def test_single_style_slerp(self):
        """1スタイルしかない場合でもslerp(t, v, v*1.05) が動作する。"""
        v = np.random.default_rng(10).standard_normal(DIM).astype(np.float32)
        v_similar = v * 1.05
        result = slerp(0.5, v, v_similar)
        # ほぼ同方向なので LERP フォールバック
        expected = lerp(0.5, v, v_similar)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_single_style_clip_norm(self):
        """参照が1ベクトルだけでもclip_normが動作する。"""
        rng = np.random.default_rng(11)
        single_ref = rng.standard_normal((1, DIM)).astype(np.float32)
        ref_norm = np.linalg.norm(single_ref[0])
        max_norm = ref_norm * 2.0

        vec = rng.standard_normal(DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec) * (max_norm * 5.0)

        result = clip_norm(vec, single_ref, max_factor=2.0)
        result_norm = np.linalg.norm(result)
        np.testing.assert_allclose(result_norm, max_norm, rtol=1e-5)
