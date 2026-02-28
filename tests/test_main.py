from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pytest
from scipy.io import wavfile

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModelHolder


def synthesize(
    inference_type: Literal["torch", "onnx"] = "torch",
    device: str = "cpu",
    onnx_providers: Sequence[tuple[str, dict[str, Any]]] = [
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ],
):

    # 音声合成モデルが配置されていれば、音声合成を実行
    model_holder = TTSModelHolder(BASE_DIR / "model_assets", device, onnx_providers)
    if len(model_holder.models_info) > 0:

        # "koharune-ami" または "amitaro" モデルを探す
        for model_info in model_holder.models_info:
            if model_info.name == "koharune-ami" or model_info.name == "amitaro":

                # Safetensors 形式または ONNX 形式のモデルファイルに絞り込む
                if inference_type == "torch":
                    model_files = [
                        f
                        for f in model_info.files
                        if f.endswith(".safetensors") and not f.startswith(".")
                    ]
                else:
                    model_files = [
                        f
                        for f in model_info.files
                        if f.endswith(".onnx") and not f.startswith(".")
                    ]
                if len(model_files) == 0:
                    pytest.skip(
                        f'音声合成モデル "{model_info.name}" のモデルファイルが見つかりませんでした。'
                    )

                # モデルをロード
                model = model_holder.get_model(model_info.name, model_files[0])
                model.load()

                # ロードされた InferenceSession の ExecutionProvider が一致するか確認
                # 一致しない場合、指定された ExecutionProvider で推論できない状態
                if inference_type == "onnx":
                    assert model.onnx_session is not None
                    assert model.onnx_session.get_providers()[0] == onnx_providers[0][0]

                # すべてのスタイルに対して音声合成を実行
                for style in model_info.styles:
                    logger.info(f"Testing style: {style}")

                    # テストに使用するサンプルテキスト
                    sample_texts = [
                        "こんにちは、初めまして。あなたの名前はなんていうの？",
                        "桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。",
                        "あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。",
                        "音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。",
                    ]

                    # 各サンプルテキストに対して音声合成を実行
                    for i, text in enumerate(sample_texts):

                        # 音声合成を実行
                        sample_rate, audio_data = model.infer(
                            text,
                            # 言語 (JP, EN, ZH / JP-Extra モデルの場合は JP のみ)
                            language=Languages.JP,
                            # 話者 ID (音声合成モデルに複数の話者が含まれる場合のみ必須、単一話者のみの場合は 0)
                            speaker_id=0,
                            # テンポの緩急 (0.0 〜 1.0)
                            sdp_ratio=0.4,
                            # スタイル (Neutral, Happy など)
                            style=style,
                            # スタイルの強さ (0.0 〜 100.0)
                            style_weight=2.0,
                        )

                        # 音声データを保存
                        (BASE_DIR / f"tests/wavs/{model_info.name}").mkdir(exist_ok=True, parents=True)  # fmt: skip
                        wav_file_path = BASE_DIR / f"tests/wavs/{model_info.name}/{style}_{i+1:02d}.wav"  # fmt: skip
                        with open(wav_file_path, "wb") as f:
                            wavfile.write(f, sample_rate, audio_data)

                        # 音声データが保存されたことを確認
                        assert wav_file_path.exists()

                # モデルをアンロード
                model.unload()
    else:
        pytest.skip("音声合成モデルが見つかりませんでした。")


def test_synthesize_cpu():
    synthesize(inference_type="torch", device="cpu")


def test_synthesize_cuda():
    synthesize(inference_type="torch", device="cuda")


def test_synthesize_onnx_cpu():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ],
    )


def test_synthesize_onnx_cuda():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "DEFAULT"}),  # fmt: skip
        ],
    )


def test_synthesize_onnx_directml():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("DmlExecutionProvider", {"device_id": 0}),
        ],
    )


def test_synthesize_onnx_coreml():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("CoreMLExecutionProvider", {}),
        ],
    )


def _get_test_model(device: str = "cpu"):
    """テスト用モデルを取得するヘルパー。モデルが見つからない場合はスキップ。"""
    onnx_providers = [
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ]
    model_holder = TTSModelHolder(BASE_DIR / "model_assets", device, onnx_providers)
    if len(model_holder.models_info) == 0:
        pytest.skip("音声合成モデルが見つかりませんでした。")

    for model_info in model_holder.models_info:
        if model_info.name in ("koharune-ami", "amitaro"):
            model_files = [
                f
                for f in model_info.files
                if f.endswith(".safetensors") and not f.startswith(".")
            ]
            if len(model_files) == 0:
                continue
            model = model_holder.get_model(model_info.name, model_files[0])
            model.load()
            return model, model_info

    pytest.skip("テスト用モデル (koharune-ami/amitaro) が見つかりませんでした。")


def test_style_vector_override_cpu():
    """style_vector_override でカスタムベクトルを直接注入して推論できることを確認"""
    model, model_info = _get_test_model("cpu")
    try:
        # モデルのスタイルベクトルから直接取得して override として渡す
        style_id = model.style2id[model_info.styles[0]]
        custom_vector = model.get_style_vector(style_id, weight=1.0)

        sample_rate, audio_data = model.infer(
            "スタイルベクトルオーバーライドのテストです。",
            language=Languages.JP,
            speaker_id=0,
            style_vector_override=custom_vector,
        )
        assert sample_rate > 0
        assert len(audio_data) > 0
    finally:
        model.unload()


def test_style_vector_override_ignores_style_param():
    """style_vector_override 指定時に style / style_weight が無視されることを確認"""
    model, model_info = _get_test_model("cpu")
    try:
        # Neutral のベクトルを override として渡しつつ、style に別の値を指定
        neutral_id = model.style2id.get("Neutral", 0)
        neutral_vector = model.get_style_vector(neutral_id, weight=1.0)

        sample_rate, audio_data = model.infer(
            "オーバーライド優先度テスト。",
            language=Languages.JP,
            speaker_id=0,
            style="Happy",  # override が優先されるため無視される
            style_weight=10.0,  # override が優先されるため無視される
            style_vector_override=neutral_vector,
        )
        assert sample_rate > 0
        assert len(audio_data) > 0
    finally:
        model.unload()


def test_style_vector_override_with_style_ops():
    """Phase 1 の style_ops 関数で生成したベクトルを override に渡せることを確認"""
    from style_bert_vits2.style_ops import slerp, validate_style_vector

    model, model_info = _get_test_model("cpu")
    try:
        styles = list(model.style2id.keys())
        vec_a = model.get_style_vector(model.style2id[styles[0]], weight=1.0)
        if len(styles) >= 2:
            vec_b = model.get_style_vector(model.style2id[styles[1]], weight=1.0)
        else:
            vec_b = vec_a * 1.1  # スタイルが1つだけの場合

        # SLERP で補間
        interpolated = slerp(0.5, vec_a, vec_b)
        validate_style_vector(interpolated)

        sample_rate, audio_data = model.infer(
            "スタイル補間のテストです。",
            language=Languages.JP,
            speaker_id=0,
            style_vector_override=interpolated,
        )
        assert sample_rate > 0
        assert len(audio_data) > 0
    finally:
        model.unload()


def test_style_vector_override_cuda():
    """CUDA での style_vector_override 推論テスト"""
    model, model_info = _get_test_model("cuda")
    try:
        style_id = model.style2id[model_info.styles[0]]
        custom_vector = model.get_style_vector(style_id, weight=1.0)

        sample_rate, audio_data = model.infer(
            "CUDAオーバーライドテスト。",
            language=Languages.JP,
            speaker_id=0,
            style_vector_override=custom_vector,
        )
        assert sample_rate > 0
        assert len(audio_data) > 0
    finally:
        model.unload()
