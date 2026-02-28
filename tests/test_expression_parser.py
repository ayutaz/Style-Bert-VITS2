"""expression_parser のユニットテスト。"""

import pytest

from style_bert_vits2.expression_parser import parse_expression

DEFAULT_STYLES = ["Happy", "Sad", "Neutral"]


class TestExpressionParser:
    """parse_expression 関数のテスト。"""

    def test_single_style(self) -> None:
        result = parse_expression("Happy", DEFAULT_STYLES)
        assert result == [(1.0, "Happy")]

    def test_two_styles_add(self) -> None:
        result = parse_expression("Happy + Sad", DEFAULT_STYLES)
        assert result == [(1.0, "Happy"), (1.0, "Sad")]

    def test_coefficient_multiply(self) -> None:
        result = parse_expression("0.5 * Happy", DEFAULT_STYLES)
        assert result == [(0.5, "Happy")]

    def test_mixed_expression(self) -> None:
        result = parse_expression("Happy - 0.3 * Neutral + 0.5 * Sad", DEFAULT_STYLES)
        assert result == [(1.0, "Happy"), (-0.3, "Neutral"), (0.5, "Sad")]

    def test_negative_first_term(self) -> None:
        result = parse_expression("-Neutral + 2.0 * Happy", DEFAULT_STYLES)
        assert result == [(-1.0, "Neutral"), (2.0, "Happy")]

    def test_subtraction(self) -> None:
        result = parse_expression("Happy - Sad", DEFAULT_STYLES)
        assert result == [(1.0, "Happy"), (-1.0, "Sad")]

    def test_all_with_coefficients(self) -> None:
        result = parse_expression(
            "0.5 * Happy + 0.3 * Sad + 0.2 * Neutral", DEFAULT_STYLES
        )
        assert result == [(0.5, "Happy"), (0.3, "Sad"), (0.2, "Neutral")]

    def test_whitespace_handling(self) -> None:
        result = parse_expression("  Happy  +  0.5 * Sad  ", DEFAULT_STYLES)
        assert result == [(1.0, "Happy"), (0.5, "Sad")]

    def test_style_name_with_hyphen(self) -> None:
        styles = DEFAULT_STYLES + ["Sad-v2"]
        result = parse_expression("Sad-v2", styles)
        assert result == [(1.0, "Sad-v2")]

    def test_style_name_with_underscore(self) -> None:
        styles = DEFAULT_STYLES + ["my_style"]
        result = parse_expression("my_style", styles)
        assert result == [(1.0, "my_style")]

    def test_empty_expression_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_expression("", DEFAULT_STYLES)

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_expression("Unknown", DEFAULT_STYLES)

    def test_no_style_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_expression("+ -", DEFAULT_STYLES)

        with pytest.raises(ValueError):
            parse_expression("0.5", DEFAULT_STYLES)

    def test_integer_coefficient(self) -> None:
        result = parse_expression("2 * Happy", DEFAULT_STYLES)
        assert result == [(2.0, "Happy")]

    def test_zero_coefficient(self) -> None:
        result = parse_expression("0 * Happy + Sad", DEFAULT_STYLES)
        assert result == [(0.0, "Happy"), (1.0, "Sad")]
