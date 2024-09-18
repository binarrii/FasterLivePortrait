import multiprocessing
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GLOG_v"] = "0"
os.environ["GLOG_logtostderr"] = "0"
os.environ["GLOG_minloglevel"] = "2"

from starlette.responses import FileResponse
from starlette.websockets import WebSocketDisconnect

import logging
import time
import cv2
import queue
import signal
import traceback

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
import mediapipe as mp
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


class VideoFramePipeline(FasterLivePortraitPipeline):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.prepare_source("aijia.png", realtime=True)
        mp_base_options = mpBaseOptions(model_asset_path=face_detect_model)
        self.face_detect_options = vision.FaceDetectorOptions(
            base_options=mp_base_options,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3
        )
        self.infer_times = []

    def handle_frame(self, frame: bytes) -> bytes:
        # noinspection PyBroadException
        try:
            if not self.src_imgs or len(self.src_imgs) <= 0:
                raise Exception("src image is empty")
            if not self.src_infos or len(self.src_infos) <= 0:
                raise Exception("src info is empty")

            t0 = time.time()
            try:
                driving_frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                dri_crop, out_crop, out_org = self.run(driving_frame,
                                                       self.src_imgs[0],
                                                       self.src_infos[0],
                                                       realtime=True)
                out_crop = cv2.cvtColor(out_crop, cv2.COLOR_BGR2RGB)
                # out_crop = stiching(cv_image, out_crop)
                if len(self.infer_times) % 15 == 0:
                    frame_rgb = cv2.cvtColor(driving_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    detector = vision.FaceDetector.create_from_options(self.face_detect_options)
                    faces = detector.detect(mp_image)
                    # noinspection PyUnresolvedReferences
                    if faces is None or len(faces.detections) <= 0:
                        self.src_lmk_pre = None
                        logging.warning("No face detected")
                        raise Exception("No face detected")
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

        is_success, buffer = cv2.imencode(".png", out_crop)
        return buffer.tobytes()


pool: queue.Queue[VideoFramePipeline] = queue.Queue(6)
for i in range(pool.maxsize):
    pool.put_nowait(VideoFramePipeline(cfg=infer_cfg))

workers = multiprocessing.Pool(processes=6)

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
        del self.active_connections[client_id]

    async def send_bytes(self, client_id: str, message: bytes):
        await self.active_connections[client_id].send_bytes(message)


connection_manager = ConnectionManager()

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse('index.html')


@app.websocket("/ws")
async def ws(websocket: WebSocket, client_id: str):
    async def handle_ws_message(client: str, message: bytes, pipe: VideoFramePipeline):
        print(f"bytes received {len(message)}")
        try:
            t0 = time.time()
            frame = pipe.handle_frame(message)
            print(f"time taken: {(time.time() - t0) * 1000}ms")

            await connection_manager.send_bytes(client, frame)
            print(f"bytes sent {len(frame)}")
        except WebSocketDisconnect:
            print("WebSocket disconnected")
            connection_manager.disconnect(client)
        except:
            traceback.print_stack()

    pipeline = pool.get()
    global terminate, workers
    try:
        await connection_manager.connect(client_id, websocket)
        while not terminate:
            data = await websocket.receive_bytes()
            workers.apply_async(func=handle_ws_message, args=(client_id, data, pipeline))
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        connection_manager.disconnect(client_id)
    finally:
        pool.put(pipeline)


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
