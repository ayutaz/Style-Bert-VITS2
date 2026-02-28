"""スタイルベクトル演算式のパーサーモジュール。

eval()を使わない安全なカスタム式パーサーを提供する。
入力文字列を ``(係数, スタイル名)`` ペアのリストに変換し、
ベクトル演算UIのカスタム式モードで使用される。

対応する文法::

    expression = term (('+' | '-') term)*
    term = [sign] [number '*'] style_name

Examples:
    >>> from style_bert_vits2.expression_parser import parse_expression
    >>> parse_expression("Happy + 0.5 * Sad", ["Happy", "Neutral", "Sad"])
    [(1.0, 'Happy'), (0.5, 'Sad')]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """トークンの種類。"""

    NUMBER = auto()
    STYLE = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    EOF = auto()


@dataclass
class Token:
    """トークン。

    Attributes:
        type: トークンの種類。
        value: トークンの文字列値。
    """

    type: TokenType
    value: str


def _tokenize(expression: str, available_styles: list[str]) -> list[Token]:
    """入力文字列をトークンのリストに分割する。

    スタイル名は available_styles リストからマッチングする。
    長い名前から先にマッチを試みることで、部分一致の問題を回避する。

    Args:
        expression: パースする式の文字列。
        available_styles: 利用可能なスタイル名のリスト。

    Returns:
        トークンのリスト。末尾にEOFトークンを含む。

    Raises:
        ValueError: 認識できない文字列が含まれている場合。
    """
    tokens: list[Token] = []
    # 長い名前から先にマッチさせる
    sorted_styles = sorted(available_styles, key=len, reverse=True)
    # スタイル名パターン (正規表現のメタ文字をエスケープ)
    style_patterns = [re.escape(s) for s in sorted_styles]

    pos = 0
    text = expression.strip()

    while pos < len(text):
        # 空白をスキップ
        if text[pos].isspace():
            pos += 1
            continue

        # 演算子
        if text[pos] == "+":
            tokens.append(Token(TokenType.PLUS, "+"))
            pos += 1
            continue

        if text[pos] == "-":
            tokens.append(Token(TokenType.MINUS, "-"))
            pos += 1
            continue

        if text[pos] == "*":
            tokens.append(Token(TokenType.MULTIPLY, "*"))
            pos += 1
            continue

        # 数値リテラル (整数部分、小数点、小数部分)
        number_match = re.match(r"\d+(?:\.\d+)?", text[pos:])
        if number_match:
            tokens.append(Token(TokenType.NUMBER, number_match.group()))
            pos += number_match.end()
            continue

        # スタイル名マッチ
        matched = False
        for pattern, style_name in zip(style_patterns, sorted_styles):
            style_match = re.match(pattern, text[pos:])
            if style_match:
                tokens.append(Token(TokenType.STYLE, style_name))
                pos += style_match.end()
                matched = True
                break

        if matched:
            continue

        # 不明な識別子を検出: 英数字+アンダースコアの連続を取得
        ident_match = re.match(r"[^\s+\-*]+", text[pos:])
        if ident_match:
            unknown = ident_match.group()
            raise ValueError(f'不明なスタイル名: "{unknown}"')

        # ここに到達するのは想定外
        raise ValueError(
            f"式の構文が不正です: 位置 {pos} で認識できない文字 '{text[pos]}'"
        )

    tokens.append(Token(TokenType.EOF, ""))
    return tokens


class _Parser:
    """トークンリストを消費して (coefficient, style_name) ペアのリストを返すパーサー。

    文法:
        expression = term (('+' | '-') term)*
        term = [sign] [number '*'] style_name
        sign = '+' | '-'
        number = float literal
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token:
        """現在位置のトークンを返す (消費しない)。"""
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        """現在位置のトークンを返して位置を進める。"""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        """指定の型のトークンを消費する。一致しない場合はエラー。"""
        token = self._peek()
        if token.type != token_type:
            raise ValueError(
                f"式の構文が不正です: {token_type.name} が期待されましたが"
                f" {token.type.name} ('{token.value}') が見つかりました"
            )
        return self._advance()

    def parse(self) -> list[tuple[float, str]]:
        """式全体をパースする。

        Returns:
            (coefficient, style_name) のタプルのリスト。

        Raises:
            ValueError: 構文エラーが検出された場合。
        """
        result = self._parse_expression()
        # 全トークンが消費されたことを確認
        if self._peek().type != TokenType.EOF:
            token = self._peek()
            raise ValueError(
                f"式の構文が不正です: 式の末尾に予期しないトークン"
                f" '{token.value}' があります"
            )
        return result

    def _parse_expression(self) -> list[tuple[float, str]]:
        """expression = term (('+' | '-') term)* をパースする。"""
        terms: list[tuple[float, str]] = []

        # 最初の term
        terms.append(self._parse_term())

        # 後続の ('+' | '-') term
        while self._peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._advance()
            coeff, style = self._parse_term()
            if op.type == TokenType.MINUS:
                coeff = -coeff
            terms.append((coeff, style))

        return terms

    def _parse_term(self) -> tuple[float, str]:
        """term = [sign] [number '*'] style_name をパースする。

        Returns:
            (coefficient, style_name) のタプル。
        """
        sign = 1.0

        # オプショナルな符号
        if self._peek().type == TokenType.PLUS:
            self._advance()
            sign = 1.0
        elif self._peek().type == TokenType.MINUS:
            self._advance()
            sign = -1.0

        # [number '*'] style_name のパターンを判定
        if self._peek().type == TokenType.NUMBER:
            number_token = self._advance()
            number_value = float(number_token.value)

            if self._peek().type == TokenType.MULTIPLY:
                # number '*' style_name
                self._advance()  # '*' を消費
                style_token = self._expect(TokenType.STYLE)
                return (sign * number_value, style_token.value)
            else:
                # 数値のあとに '*' がない場合は構文エラー
                raise ValueError(
                    f"式の構文が不正です: 数値 {number_token.value} の後に"
                    f" '*' とスタイル名が必要です"
                )

        elif self._peek().type == TokenType.STYLE:
            # style_name のみ (係数なし)
            style_token = self._advance()
            return (sign * 1.0, style_token.value)

        else:
            token = self._peek()
            raise ValueError(
                f"式の構文が不正です: 数値またはスタイル名が期待されましたが"
                f" '{token.value}' が見つかりました"
            )


def parse_expression(
    expression: str, available_styles: list[str]
) -> list[tuple[float, str]]:
    """スタイルベクトル演算式をパースする。

    安全な（eval()を使わない）カスタム式パーサー。
    入力文字列を (係数, スタイル名) ペアのリストに変換する。

    文法::

        expression = term (('+' | '-') term)*
        term = [sign] [number '*'] style_name
        sign = '+' | '-'
        number = float literal
        style_name = available_styles のいずれかにマッチする文字列

    Args:
        expression: パースする式の文字列。
            例: ``"Happy - 0.3 * Neutral + 0.5 * Sad"``
        available_styles: 利用可能なスタイル名のリスト。
            例: ``["Happy", "Neutral", "Sad"]``

    Returns:
        (係数, スタイル名) のタプルのリスト。
        例: ``[(1.0, "Happy"), (-0.3, "Neutral"), (0.5, "Sad")]``

    Raises:
        ValueError: 式が空、未知のスタイル名、構文エラー、
            またはスタイル名が1つも含まれない場合。

    Examples:
        >>> parse_expression("Happy", ["Happy", "Neutral", "Sad"])
        [(1.0, 'Happy')]

        >>> parse_expression("0.5 * Happy + 0.5 * Sad", ["Happy", "Neutral", "Sad"])
        [(0.5, 'Happy'), (0.5, 'Sad')]

        >>> parse_expression("Happy - 0.3 * Neutral", ["Happy", "Neutral", "Sad"])
        [(1.0, 'Happy'), (-0.3, 'Neutral')]
    """
    if not expression or not expression.strip():
        raise ValueError("式が空です。")

    tokens = _tokenize(expression, available_styles)

    # スタイル名が1つも含まれていないかチェック
    has_style = any(t.type == TokenType.STYLE for t in tokens)
    if not has_style:
        raise ValueError("式にスタイル名が含まれていません。")

    parser = _Parser(tokens)
    return parser.parse()
