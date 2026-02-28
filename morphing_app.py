"""HTMXベースのモーフィングUIバックエンドAPIルーター。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.style_ops import (
    clip_norm,
    compute_norm_ratio,
    lerp,
    load_style_vectors,
    save_style_vectors,
    slerp,
    validate_style_vector,
)

router = APIRouter()

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_morphing_state: dict = {
    "model_name": None,
    "model_path": None,
    "vectors": None,  # np.ndarray (num_styles, 256)
    "style2id": None,  # dict[str, int]
}


@router.get("/morphing", response_class=HTMLResponse)
async def morphing_page(request: Request):
    """メインページ (morphing.html)。"""
    model_holder = request.app.state.model_holder
    model_names = model_holder.model_names
    return templates.TemplateResponse(
        "morphing.html",
        {"request": request, "model_names": model_names},
    )


@router.get("/api/morphing/model-files", response_class=HTMLResponse)
async def get_model_files(request: Request, model_name: str):
    """指定モデルの .safetensors ファイルを option タグで返す。"""
    model_holder = request.app.state.model_holder
    # models_info からモデルを探す
    options_html = ""
    for info in model_holder.models_info:
        if info.name == model_name:
            for f in info.files:
                if f.endswith(".safetensors"):
                    file_name = Path(f).name
                    options_html += f'<option value="{f}">{file_name}</option>\n'
            break
    return HTMLResponse(content=options_html)


@router.post("/api/morphing/load-model", response_class=HTMLResponse)
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

    _morphing_state["model_name"] = model_name
    _morphing_state["model_path"] = model_path
    _morphing_state["vectors"] = vectors
    _morphing_state["style2id"] = style2id

    styles = list(style2id.keys())
    return templates.TemplateResponse(
        "partials/style_options.html",
        {"request": request, "styles": styles},
    )


@router.post("/api/morphing/compute-norm", response_class=HTMLResponse)
async def compute_norm(
    request: Request,
    style_a: str = Form(...),
    style_b: str = Form(...),
    method: str = Form(...),
    ratio: float = Form(...),
):
    """補間結果のノルム比を計算して返す。"""
    vectors = _morphing_state["vectors"]
    style2id = _morphing_state["style2id"]

    if vectors is None or style2id is None:
        return HTMLResponse(
            content='<div class="error">モデルをロードしてください</div>'
        )

    vec_a = vectors[style2id[style_a]]
    vec_b = vectors[style2id[style_b]]

    if method == "slerp":
        result = slerp(ratio, vec_a, vec_b)
    else:
        result = lerp(ratio, vec_a, vec_b)

    norm_ratio = compute_norm_ratio(result, vectors)

    return templates.TemplateResponse(
        "partials/norm_indicator.html",
        {"request": request, "norm_ratio": norm_ratio},
    )


@router.post("/api/morphing/synthesize", response_class=HTMLResponse)
async def synthesize(
    request: Request,
    style_a: str = Form(...),
    style_b: str = Form(...),
    method: str = Form(...),
    ratio: float = Form(...),
    text: str = Form(...),
    language: str = Form(...),
    speaker_id: int = Form(...),
):
    """補間ベクトルで音声合成し、base64エンコードした音声を返す。"""
    vectors = _morphing_state["vectors"]
    style2id = _morphing_state["style2id"]
    model_name = _morphing_state["model_name"]
    model_path = _morphing_state["model_path"]

    if vectors is None or style2id is None:
        return HTMLResponse(
            content='<div class="error">モデルをロードしてください</div>'
        )

    vec_a = vectors[style2id[style_a]]
    vec_b = vectors[style2id[style_b]]

    if method == "slerp":
        interpolated = slerp(ratio, vec_a, vec_b)
    else:
        interpolated = lerp(ratio, vec_a, vec_b)

    try:
        validate_style_vector(interpolated)
    except ValueError as e:
        return HTMLResponse(content=f'<div class="error">{e}</div>')

    model_holder = request.app.state.model_holder
    model = model_holder.get_model(model_name, model_path)

    sr, audio = model.infer(
        text=text,
        language=Languages(language),
        speaker_id=speaker_id,
        style_vector_override=interpolated,
    )

    buf = BytesIO()
    scipy.io.wavfile.write(buf, sr, audio)
    audio_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    norm_ratio = compute_norm_ratio(interpolated, vectors)

    return templates.TemplateResponse(
        "partials/audio_player.html",
        {"request": request, "audio_data": audio_data, "norm_ratio": norm_ratio},
    )


@router.post("/api/morphing/save-style", response_class=HTMLResponse)
async def save_style(
    request: Request,
    style_a: str = Form(...),
    style_b: str = Form(...),
    method: str = Form(...),
    ratio: float = Form(...),
    style_name: str = Form(...),
):
    """補間結果を新スタイルとして保存する。"""
    vectors = _morphing_state["vectors"]
    style2id = _morphing_state["style2id"]
    model_name = _morphing_state["model_name"]

    if vectors is None or style2id is None:
        return HTMLResponse(
            content='<div class="error">モデルをロードしてください</div>'
        )

    if style_name in style2id:
        return HTMLResponse(
            content=f'<div class="error">スタイル名 "{style_name}" は既に存在します</div>'
        )

    vec_a = vectors[style2id[style_a]]
    vec_b = vectors[style2id[style_b]]

    if method == "slerp":
        result = slerp(ratio, vec_a, vec_b)
    else:
        result = lerp(ratio, vec_a, vec_b)

    result = clip_norm(result, vectors)

    new_vectors = np.vstack([vectors, result.reshape(1, -1)])
    new_style2id = dict(style2id)
    new_style2id[style_name] = len(style2id)

    model_holder = request.app.state.model_holder
    save_style_vectors(new_vectors, new_style2id, model_name, model_holder.root_dir)

    _morphing_state["vectors"] = new_vectors
    _morphing_state["style2id"] = new_style2id

    return HTMLResponse(
        content=f'<div class="success">スタイル "{style_name}" を保存しました</div>'
    )
