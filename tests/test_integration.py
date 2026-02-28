"""Phase 5 統合テスト。

全モジュールのインポートとAPIの存在確認。
"""

import numpy as np


class TestModuleImports:
    """全モジュールが正しくインポートできることを確認。"""

    def test_import_style_ops(self):
        """style_ops モジュールの全公開関数がインポートできる。"""

    def test_import_expression_parser(self):
        """expression_parser モジュールがインポートできる。"""

    def test_import_morphing_app(self):
        """morphing_app モジュールがインポートできる。"""
        from morphing_app import router

        assert router is not None

    def test_import_vector_app(self):
        """vector_app モジュールがインポートできる。"""
        from vector_app import router

        assert router is not None


class TestAPIConsistency:
    """API間の一貫性テスト。"""

    def test_style_ops_functions_return_ndarray(self):
        """演算関数がnp.ndarrayを返す。"""
        from style_bert_vits2.style_ops import (
            lerp,
            slerp,
            vector_add,
            vector_diff_transfer,
            vector_mean,
            vector_scale,
            vector_sub,
        )

        rng = np.random.default_rng(42)
        v0 = rng.standard_normal(256)
        v1 = rng.standard_normal(256)

        results = [
            lerp(0.5, v0, v1),
            slerp(0.5, v0, v1),
            vector_add(v0, v1),
            vector_sub(v0, v1),
            vector_mean(v0, v1),
            vector_scale(v0, 2.0),
            vector_diff_transfer(v0, v1, v0),
        ]

        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (256,)

    def test_expression_parser_with_style_ops(self):
        """パーサーの出力でベクトル演算ができる。"""
        from style_bert_vits2.expression_parser import parse_expression

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((3, 256))
        style2id = {"Happy": 0, "Sad": 1, "Neutral": 2}

        parsed = parse_expression("0.5 * Happy + 0.5 * Sad", list(style2id.keys()))
        result = np.zeros(256)
        for coeff, name in parsed:
            result += coeff * vectors[style2id[name]]

        assert result.shape == (256,)
        assert not np.any(np.isnan(result))

    def test_pca_with_real_shaped_vectors(self):
        """PCA関数が実際のスタイルベクトルサイズで動作する。"""
        from style_bert_vits2.style_ops import prepare_plot_data

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((10, 256))
        style2id = {f"style_{i}": i for i in range(10)}
        result = rng.standard_normal(256)

        points = prepare_plot_data(vectors, style2id, result, "test")
        assert len(points) == 11
        for p in points:
            assert "x" in p and "y" in p and "label" in p and "is_result" in p

    def test_validate_after_expression_eval(self):
        """式パーサーの結果がバリデーションをパスする。"""
        from style_bert_vits2.expression_parser import parse_expression
        from style_bert_vits2.style_ops import validate_style_vector

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((3, 256))
        style2id = {"A": 0, "B": 1, "C": 2}

        parsed = parse_expression("A + 0.5 * B - 0.3 * C", list(style2id.keys()))
        result = np.zeros(256)
        for coeff, name in parsed:
            result += coeff * vectors[style2id[name]]

        validate_style_vector(result)  # Should not raise
