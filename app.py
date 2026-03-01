import argparse
import threading
import webbrowser
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import get_path_config
from inference_app import router as inference_router
from morphing_app import router as morphing_router
from style_bert_vits2.constants import BASE_DIR
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

app = FastAPI(title="Style-Bert-VITS2")
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)
app.state.model_holder = model_holder
app.include_router(inference_router)
app.include_router(morphing_router)
app.include_router(vector_router)


@app.get("/")
async def root():
    """ルートパスを推論UIにリダイレクト。"""
    return RedirectResponse(url="/inference")


if not args.no_autolaunch:
    threading.Timer(
        1.5,
        lambda: webbrowser.open(f"http://127.0.0.1:{args.port}"),
    ).start()

uvicorn.run(app, host=args.host, port=args.port)
