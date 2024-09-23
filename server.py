import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GLOG_v"] = "0"
os.environ["GLOG_logtostderr"] = "0"
os.environ["GLOG_minloglevel"] = "2"

import asyncio
import logging
import time
import cv2
import queue
import requests
import signal
import traceback
import threading
import uuid
import torch
import uvicorn

import mediapipe as mp
import numpy as np

from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.websockets import WebSocketDisconnect
from starlette import status
from PIL import Image
from mediapipe.tasks.python import BaseOptions as mpBaseOptions
from mediapipe.tasks.python import vision
from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

logger = logging.getLogger(__file__)

face_detect_model = "checkpoints/liveportrait_onnx/face_detector.tflite"
conf_file = "configs/onnx_infer.yaml"
if torch.cuda.is_available():
    conf_file = "configs/trt_mp_infer.yaml"

infer_cfg = OmegaConf.load(conf_file)
infer_cfg.infer_params.flag_pasteback = False

lock = threading.Lock()

class VideoFramePipeline(FasterLivePortraitPipeline):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        mp_base_options = mpBaseOptions(model_asset_path=face_detect_model)
        face_detect_options = vision.FaceDetectorOptions(
            base_options=mp_base_options,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3
        )
        self.face_detector = vision.FaceDetector.create_from_options(face_detect_options)
        self.faces = None
        self.infer_times = []

    def prepare_source(self, source_path, **kwargs):
        self.src_infos = []
        self.src_imgs = []
        self.src_lmk_pre = None
        self.infer_times = []
        return super().prepare_source(source_path, **kwargs)

    def handle_frame(self, frame: bytes) -> bytes:
        # noinspection PyBroadException
        driving_frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        driving_frame = cv2.flip(driving_frame, 1)
        try:
            if not self.src_imgs or len(self.src_imgs) <= 0:
                raise Exception("src image is empty")
            if not self.src_infos or len(self.src_infos) <= 0:
                raise Exception("src info is empty")

            t0 = time.time()
            try:
                frame_rgb = cv2.cvtColor(driving_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                faces = self.face_detector.detect(mp_image)
                if faces is None or len(faces.detections) <= 0:
                    self.src_lmk_pre = None
                    raise Exception("No face detected")
                
                dri_crop, out_crop, out_org = self.run(driving_frame, self.src_imgs[0],
                                                       self.src_infos[0], realtime=True, faces=faces)
                new_size = (out_crop.shape[1] * 2, out_crop.shape[0] * 2)
                out_crop = cv2.resize(out_crop, new_size, interpolation=cv2.INTER_LANCZOS4)
                out_crop = cv2.cvtColor(out_crop, cv2.COLOR_BGR2RGB)
            finally:
                if len(self.infer_times) > 7200:
                    self.infer_times.pop(0)
                self.infer_times.append(time.time() - t0)

            if len(self.infer_times) % 60 == 0:
                print(f"inference median time: {np.median(self.infer_times) * 1000} ms/frame, "
                      f"mean time: {np.mean(self.infer_times) * 1000} ms/frame")
        except Exception as e:
            if len(self.infer_times) % 60 == 0:
                logging.warning(f"{repr(e)}")
            out_crop = driving_frame

        is_success, buffer = cv2.imencode(".jpeg", out_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()


terminate = False


def signal_handler(sig, frame):
    global terminate
    terminate = True


signal.signal(signal.SIGINT, signal_handler)


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_bytes(self, client_id: str, message: bytes):
        await self.active_connections[client_id].send_bytes(message)


pool: queue.Queue[VideoFramePipeline] = queue.Queue(6)
for i in range(pool.maxsize):
    pool.put_nowait(VideoFramePipeline(cfg=infer_cfg))

connection_manager = ConnectionManager()
clients: dict[str, VideoFramePipeline] = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/portraits", StaticFiles(directory="portraits"), name="portraits")
app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")
app.mount("/images", StaticFiles(directory="dist/images"), name="images")


@app.get("/")
async def index():
    return FileResponse('dist/index.html')


@app.get("/apt/getportraits")
@app.get("/getportraits")
async def get_portrait(request: Request):
    portraits = []
    for f in os.listdir("portraits"):
        if f.startswith('t_'):
            continue
        name = f.split(".")[0]
        portraits.append({"name": name, "url": f"/portraits/{f}"})
    return portraits


@app.post("/apt/setportrait")
@app.post("/setportrait")
async def set_portrait(request: Request):
    json = await request.json()
    client_id = json.get("client_id", None)
    portrait = json.get("portrait", None)
    if not client_id or not portrait:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    portrait_path = f"portraits/{portrait}.png"
    if portrait.startswith('http'):
        img = requests.get(portrait).content
        portrait = f"t_{uuid.uuid4().hex}"
        with open(f"/tmp/{portrait}.png", 'wb') as f:
            f.write(img)
        portrait_path = f"/tmp/{portrait}.png"

    client_id = str(client_id)
    pipeline = clients.get(client_id, None)
    if pipeline is not None:
        with lock:
            pipeline.prepare_source(portrait_path, realtime=True)
            logger.info(f"Portrait changed: {client_id} -> {portrait}")
    else:
        logger.error(f"Invalid client: {client_id}")


@app.websocket("/ws")
async def ws(websocket: WebSocket, client_id: str, portrait: str = "aijia"):
    if not client_id or not portrait:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    async def handle_ws_message(client: str, data: bytes, pipe: VideoFramePipeline):
        # print(f"bytes received {len(data)}")
        try:
            # t0 = time.time()
            frame = pipe.handle_frame(data)
            # print(f"time taken: {(time.time() - t0) * 1000}ms")

            await connection_manager.send_bytes(client, frame)
            # print(f"bytes sent {len(frame)}")
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {client_id}")
            connection_manager.disconnect(client)
        except:
            traceback.print_stack()

    client_id = str(client_id)
    pipeline = clients.get(client_id, None)
    try:
        if pipeline is None:
            pipeline = pool.get_nowait()
            clients[client_id] = pipeline

        if pipeline is None:
            return

        with lock:
            pipeline.prepare_source(f"portraits/{portrait}.png", realtime=True)

        await connection_manager.connect(client_id, websocket)
        global terminate
        while not terminate:
            message = await websocket.receive_bytes()
            asyncio.ensure_future(handle_ws_message(client_id, message, pipeline))
    except queue.Empty:
        logger.warning(f"No pipeline available: {client_id}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
    finally:
        connection_manager.disconnect(client_id)
        pool.put(pipeline)
        logger.info(f"Pipeline recycled: {client_id}")
        if client_id in clients:
            del clients[client_id]
            logger.info(f"Client removed: {client_id}")


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=9090, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
