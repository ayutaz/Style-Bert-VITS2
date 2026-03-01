"""HTMXベースの推論UIバックエンドAPIルーター。

テキストから音声を合成する推論機能を提供する。
FastAPI APIRouterとして実装され、app.pyからマウントされる。

Note:
    ``_inference_state`` モジュール変数でセッション中の状態（読み込み済み
    モデル名等）を保持する。シングルユーザー前提の設計。
"""

from __future__ import annotations

import base64
from io import BytesIO

import scipy.io.wavfile
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.logging import logger


router = APIRouter()

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_inference_state: dict = {
    "model_name": None,
    "model_path": None,
}


@router.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request):
    """メインページ (inference.html)。"""
    model_holder = request.app.state.model_holder
    model_names = model_holder.model_names
    return templates.TemplateResponse(
        "inference.html",
        {"request": request, "model_names": model_names},
    )


@router.get("/api/inference/model-files", response_class=HTMLResponse)
async def get_model_files(request: Request, model_name: str):
    """指定モデルの .safetensors ファイルを option タグで返す。

    Args:
        request: FastAPIリクエストオブジェクト。
        model_name: モデル名（クエリパラメータ）。

    Returns:
        ``<option>`` タグを連結したHTMLフラグメント。
    """
    from pathlib import Path

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


@router.post("/api/inference/load-model", response_class=HTMLResponse)
async def load_model(
    request: Request,
    model_name: str = Form(...),
    model_path: str = Form(...),
):
    """モデルをロードし、スタイル・話者選択肢を返す。

    Args:
        request: FastAPIリクエストオブジェクト。
        model_name: モデル名。
        model_path: モデルファイル（.safetensors）のパス。

    Returns:
        スタイル・話者選択肢を含むHTML部分テンプレート。
        読み込み失敗時はエラーメッセージのHTMLを返す。
    """
    model_holder = request.app.state.model_holder
    try:
        model = model_holder.get_model(model_name, model_path)
    except Exception as e:
        return HTMLResponse(content=f'<div class="error">{e}</div>')

    _inference_state["model_name"] = model_name
    _inference_state["model_path"] = model_path

    styles = list(model.style2id.keys())
    speakers = list(model.spk2id.keys())
    return templates.TemplateResponse(
        "partials/inference_options.html",
        {"request": request, "styles": styles, "speakers": speakers},
    )


@router.post("/api/inference/synthesize", response_class=HTMLResponse)
async def synthesize(
    request: Request,
    model_name: str = Form(...),
    model_path: str = Form(...),
    text: str = Form(...),
    language: str = Form(...),
    speaker: str = Form(...),
    style: str = Form(...),
    style_weight: float = Form(1.0),
    sdp_ratio: float = Form(0.2),
    noise: float = Form(0.6),
    noise_w: float = Form(0.8),
    length: float = Form(1.0),
    line_split: str | None = Form(None),
    split_interval: float = Form(0.5),
    pitch_scale: float = Form(1.0),
    intonation_scale: float = Form(1.0),
    use_assist_text: str | None = Form(None),
    assist_text: str = Form(""),
    assist_text_weight: float = Form(1.0),
):
    """テキストから音声を合成し、base64エンコードした音声を返す。

    Args:
        request: FastAPIリクエストオブジェクト。
        model_name: モデル名。
        model_path: モデルファイルのパス。
        text: 合成するテキスト。
        language: 言語コード（JP/EN/ZH）。
        speaker: 話者名。
        style: スタイル名。
        style_weight: スタイルの重み。
        sdp_ratio: SDP比率。
        noise: ノイズスケール。
        noise_w: ノイズスケールW。
        length: 長さスケール。
        line_split: 改行分割（チェックボックス、"on"または None）。
        split_interval: 分割時の無音秒数。
        pitch_scale: ピッチスケール。
        intonation_scale: 抑揚スケール。
        use_assist_text: 補助テキスト使用（チェックボックス、"on"または None）。
        assist_text: 補助テキスト。
        assist_text_weight: 補助テキストの重み。

    Returns:
        オーディオプレイヤーを含むHTML部分テンプレート。
        エラー時はエラーメッセージのHTMLを返す。
    """
    line_split_bool = line_split is not None
    use_assist_text_bool = use_assist_text is not None

    model_holder = request.app.state.model_holder
    try:
        model = model_holder.get_model(model_name, model_path)
        speaker_id = model.spk2id[speaker]

        sr, audio = model.infer(
            text=text,
            language=Languages(language),
            speaker_id=speaker_id,
            style=style,
            style_weight=style_weight,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noise_w,
            length=length,
            line_split=line_split_bool,
            split_interval=split_interval,
            pitch_scale=pitch_scale,
            intonation_scale=intonation_scale,
            use_assist_text=use_assist_text_bool,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            force_reload_model=False,
        )
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return HTMLResponse(content=f'<div class="error">{e}</div>')

    buf = BytesIO()
    scipy.io.wavfile.write(buf, sr, audio)
    audio_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return templates.TemplateResponse(
        "partials/audio_player.html",
        {"request": request, "audio_data": audio_data},
    )


@router.post("/api/inference/refresh-models", response_class=HTMLResponse)
async def refresh_models(request: Request):
    """モデル一覧を更新し、新しいモデル名のoptionタグを返す。

    Args:
        request: FastAPIリクエストオブジェクト。

    Returns:
        ``<option>`` タグを連結したHTMLフラグメント。
    """
    model_holder = request.app.state.model_holder
    model_holder.refresh()
    options_html = ""
    for name in model_holder.model_names:
        options_html += f'<option value="{name}">{name}</option>\n'
    return HTMLResponse(content=options_html)
