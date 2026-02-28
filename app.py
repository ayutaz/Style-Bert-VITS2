import argparse
import threading
import webbrowser
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import get_path_config
from gradio_tabs.inference import create_inference_app
from morphing_app import router as morphing_router
from style_bert_vits2.constants import BASE_DIR, GRADIO_THEME, VERSION
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers
from vector_app import router as vector_router

# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--no_autolaunch", action="store_true")
parser.add_argument("--share", action="store_true")

args = parser.parse_args()
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

path_config = get_path_config()
model_holder = TTSModelHolder(
    Path(path_config.assets_root),
    device,
    torch_device_to_onnx_providers(device),
    ignore_onnx=True,
)

# Gradio UI の構築
with gr.Blocks(theme=GRADIO_THEME) as gradio_app:
    gr.Markdown(f"# Style-Bert-VITS2 WebUI (version {VERSION})")
    with gr.Row():
        gr.Markdown(
            "[モーフィング UI を開く](/morphing) | [ベクトル演算 UI を開く](/vector-arithmetic)"
        )
    create_inference_app(model_holder=model_holder)

if args.share:
    # --share モード: Gradio のトンネル機能を使用（HTMX UIは利用不可）
    gradio_app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_autolaunch,
        share=True,
    )
else:
    # 通常モード: FastAPI + Gradio mount + HTMX
    fastapi_app = FastAPI(title="Style-Bert-VITS2")
    fastapi_app.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )
    fastapi_app.state.model_holder = model_holder
    fastapi_app.include_router(morphing_router)
    fastapi_app.include_router(vector_router)
    fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/")

    if not args.no_autolaunch:
        threading.Timer(
            1.5,
            lambda: webbrowser.open(f"http://127.0.0.1:{args.port}"),
        ).start()

    uvicorn.run(fastapi_app, host=args.host, port=args.port)
