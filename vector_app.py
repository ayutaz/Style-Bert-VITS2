"""HTMXベースのベクトル演算UIバックエンドAPIルーター。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.expression_parser import parse_expression
from style_bert_vits2.style_ops import (
    clip_norm,
    compute_norm_ratio,
    load_style_vectors,
    save_style_vectors,
    validate_style_vector,
    vector_diff_transfer,
    vector_mean,
    vector_scale,
)

router = APIRouter()

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_vector_state: dict = {
    "model_name": None,
    "model_path": None,
    "vectors": None,  # np.ndarray (num_styles, 256)
    "style2id": None,  # dict[str, int]
}


def _compute_result_vector(
    form: dict, vectors: np.ndarray, style2id: dict
) -> np.ndarray:
    """フォームデータからモードに応じたベクトル演算を実行する。

    Args:
        form: フォームデータの辞書。
        vectors: スタイルベクトル配列。shape (num_styles, 256)。
        style2id: スタイル名からインデックスへのマッピング。

    Returns:
        演算結果ベクトル。shape (256,)。

    Raises:
        ValueError: 不正なモードや式が指定された場合。
    """
    mode = form.get("mode")

    if mode == "diff_transfer":
        base = vectors[style2id[form["dt_base"]]]
        source = vectors[style2id[form["dt_source"]]]
        target = vectors[style2id[form["dt_target"]]]
        scale = float(form.get("dt_scale", "1.0"))
        return vector_diff_transfer(base, source, target, scale)

    elif mode == "weighted_mean":
        style_a = form["wm_style_a"]
        weight_a = float(form["wm_weight_a"])
        style_b = form["wm_style_b"]
        weight_b = float(form["wm_weight_b"])

        styles_weights = [
            (weight_a, vectors[style2id[style_a]]),
            (weight_b, vectors[style2id[style_b]]),
        ]

        style_c = form.get("wm_style_c")
        if style_c and style_c.strip():
            weight_c = float(form.get("wm_weight_c", "0.0"))
            styles_weights.append((weight_c, vectors[style2id[style_c]]))

        normalize = form.get("wm_normalize") == "on"
        weights = [w for w, _ in styles_weights]
        vecs = [v for _, v in styles_weights]

        result = sum(w * v for w, v in zip(weights, vecs))
        if normalize:
            weight_sum = sum(weights)
            if weight_sum != 0.0:
                result = result / weight_sum

        return result

    elif mode == "scaling":
        vec = vectors[style2id[form["sc_style"]]]
        scale = float(form.get("sc_scale", "1.0"))
        return vector_scale(vec, scale)

    elif mode == "custom":
        expression = form["expression"]
        parsed = parse_expression(expression, list(style2id.keys()))
        result = np.zeros(vectors.shape[1], dtype=vectors.dtype)
        for coeff, name in parsed:
            result = result + coeff * vectors[style2id[name]]
        return result

    else:
        raise ValueError(f"不明なモード: {mode}")


@router.get("/vector-arithmetic", response_class=HTMLResponse)
async def vector_arithmetic_page(request: Request):
    """メインページ (vector_arithmetic.html)。"""
    model_holder = request.app.state.model_holder
    model_names = model_holder.model_names
    return templates.TemplateResponse(
        "vector_arithmetic.html",
        {"request": request, "model_names": model_names},
    )


@router.get("/api/vector/model-files", response_class=HTMLResponse)
async def get_model_files(request: Request, model_name: str):
    """指定モデルの .safetensors ファイルを option タグで返す。"""
    model_holder = request.app.state.model_holder
    options_html = ""
    for info in model_holder.models_info:
        if info.name == model_name:
            for f in info.files:
                if f.endswith(".safetensors"):
                    file_name = Path(f).name
                    options_html += f'<option value="{f}">{file_name}</option>\n'
            break
    return HTMLResponse(content=options_html)


@router.post("/api/vector/load-model", response_class=HTMLResponse)
async def load_model(
    request: Request,
    model_name: str = Form(...),
    model_path: str = Form(...),
):
    """モデルのスタイルベクトルを読み込み、スタイル選択肢を返す。"""
    model_holder = request.app.state.model_holder
    try:
        vectors, style2id = load_style_vectors(model_name, model_holder.root_dir)
    except Exception as e:
        return HTMLResponse(content=f'<div class="error">{e}</div>')

    _vector_state["model_name"] = model_name
    _vector_state["model_path"] = model_path
    _vector_state["vectors"] = vectors
    _vector_state["style2id"] = style2id

    styles = list(style2id.keys())
    return templates.TemplateResponse(
        "partials/va_style_options.html",
        {"request": request, "styles": styles},
    )


@router.post("/api/vector/compute-norm", response_class=HTMLResponse)
async def compute_norm(request: Request):
    """演算結果のノルム比を計算して返す。"""
    vectors = _vector_state["vectors"]
    style2id = _vector_state["style2id"]

    if vectors is None or style2id is None:
        return HTMLResponse(
            content='<div class="error">モデルをロードしてください</div>'
        )

    try:
        form = await request.form()
        form_dict = dict(form)
        result = _compute_result_vector(form_dict, vectors, style2id)

        if form_dict.get("clip_enabled") == "on":
            clip_factor = float(form_dict.get("clip_factor", "2.0"))
            result = clip_norm(result, vectors, clip_factor)

        norm_ratio = compute_norm_ratio(result, vectors)
    except (ValueError, KeyError) as e:
        return HTMLResponse(content=f'<div class="error">{e}</div>')

    return templates.TemplateResponse(
        "partials/norm_indicator.html",
        {"request": request, "norm_ratio": norm_ratio},
    )


@router.post("/api/vector/synthesize", response_class=HTMLResponse)
async def synthesize(request: Request):
    """演算結果ベクトルで音声合成し、base64エンコードした音声を返す。"""
    vectors = _vector_state["vectors"]
    style2id = _vector_state["style2id"]
    model_name = _vector_state["model_name"]
    model_path = _vector_state["model_path"]

    if vectors is None or style2id is None:
        return HTMLResponse(
            content='<div class="error">モデルをロードしてください</div>'
        )

    try:
        form = await request.form()
        form_dict = dict(form)
        result = _compute_result_vector(form_dict, vectors, style2id)

        if form_dict.get("clip_enabled") == "on":
            clip_factor = float(form_dict.get("clip_factor", "2.0"))
            result = clip_norm(result, vectors, clip_factor)

        validate_style_vector(result)

        text = form_dict["text"]
        language = form_dict["language"]
        speaker_id = int(form_dict["speaker_id"])

        model_holder = request.app.state.model_holder
        model = model_holder.get_model(model_name, model_path)

        sr, audio = model.infer(
            text=text,
            language=Languages(language),
            speaker_id=speaker_id,
            style_vector_override=result,
        )

        buf = BytesIO()
        scipy.io.wavfile.write(buf, sr, audio)
        audio_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        norm_ratio = compute_norm_ratio(result, vectors)
    except (ValueError, KeyError) as e:
        return HTMLResponse(content=f'<div class="error">{e}</div>')

    return templates.TemplateResponse(
        "partials/audio_player.html",
        {"request": request, "audio_data": audio_data, "norm_ratio": norm_ratio},
    )


@router.post("/api/vector/save-style", response_class=HTMLResponse)
async def save_style(request: Request):
    """演算結果を新スタイルとして保存する。"""
    vectors = _vector_state["vectors"]
    style2id = _vector_state["style2id"]
    model_name = _vector_state["model_name"]

    if vectors is None or style2id is None:
        return HTMLResponse(
            content='<div class="error">モデルをロードしてください</div>'
        )

    try:
        form = await request.form()
        form_dict = dict(form)
        style_name = form_dict["style_name"]

        if style_name in style2id:
            return HTMLResponse(
                content=f'<div class="error">スタイル名 "{style_name}" は既に存在します</div>'
            )

        result = _compute_result_vector(form_dict, vectors, style2id)

        # 保存時は常にclip_normを適用
        clip_factor = float(form_dict.get("clip_factor", "2.0"))
        result = clip_norm(result, vectors, clip_factor)

        new_vectors = np.vstack([vectors, result.reshape(1, -1)])
        new_style2id = dict(style2id)
        new_style2id[style_name] = len(style2id)

        model_holder = request.app.state.model_holder
        save_style_vectors(new_vectors, new_style2id, model_name, model_holder.root_dir)

        _vector_state["vectors"] = new_vectors
        _vector_state["style2id"] = new_style2id
    except (ValueError, KeyError) as e:
        return HTMLResponse(content=f'<div class="error">{e}</div>')

    return HTMLResponse(
        content=f'<div class="success">スタイル "{style_name}" を保存しました</div>'
    )
