import os

from starlette.responses import FileResponse
from starlette.websockets import WebSocketDisconnect

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GLOG_v"] = "0"
os.environ["GLOG_logtostderr"] = "0"
os.environ["GLOG_minloglevel"] = "2"

import av

av.logging.set_level(av.logging.FATAL)
av.logging.set_libav_level(av.logging.FATAL)

import io
import logging
import time
import cv2
import signal
import traceback

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, WebSocket
import numpy as np
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
pipe = FasterLivePortraitPipeline(cfg=infer_cfg)
pipe.prepare_source("aijia.png", realtime=True)

mp_base_options = mpBaseOptions(model_asset_path=face_detect_model)
face_detect_options = vision.FaceDetectorOptions(
    base_options=mp_base_options,
    min_detection_confidence=0.5,
    min_suppression_threshold=0.3
)

infer_times = []


def hand_frame(frame: av.VideoFrame) -> av.VideoFrame:
    # Debug #
    # print(f"w: {frame.width}, h: {frame.height}")
    # print(f"format: {frame.format}")
    # print(f"time_base: {frame.time_base}, time: {frame.time}")
    # print(f"pts: {frame.pts}")
    # print(f"dts: {frame.dts}")
    # print(f"colorspace: {frame.colorspace}")
    # print(f"color_range: {frame.color_range}")
    # print(f"pict_type: {frame.pict_type}")
    # print(f"planes: {len(frame.planes)}")
    # print(f"is_corrupt: {frame.is_corrupt}")
    # Debug #

    driving_frame = frame.to_ndarray(format="bgr24")
    # noinspection PyBroadException
    try:
        if not pipe.src_imgs or len(pipe.src_imgs) <= 0:
            raise Exception("src image is empty")
        if not pipe.src_infos or len(pipe.src_infos) <= 0:
            raise Exception("src info is empty")

        t0 = time.time()
        try:
            dri_crop, out_crop, out_org = pipe.run(driving_frame, pipe.src_imgs[0], pipe.src_infos[0],
                                                   realtime=True)
            out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
            # out_crop = stiching(cv_image, out_crop)
            if len(infer_times) % 15 == 0:
                frame_rgb = cv2.cvtColor(driving_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                faces = vision.FaceDetector.create_from_options(face_detect_options).detect(mp_image)
                # noinspection PyUnresolvedReferences
                if faces is None or len(faces.detections) <= 0:
                    pipe.src_lmk_pre = None
                    logging.warning("No face detected")
                    raise Exception("No face detected")
        finally:
            if len(infer_times) > 7200:
                infer_times.pop(0)
            infer_times.append(time.time() - t0)

        if len(infer_times) % 60 == 0:
            print(f"inference median time: {np.median(infer_times) * 1000} ms/frame, "
                  f"mean time: {np.mean(infer_times) * 1000} ms/frame")
    except Exception as e:
        if len(infer_times) % 60 == 0:
            logging.warning(f"{repr(e)}")
        out_crop = driving_frame

    return av.VideoFrame.from_ndarray(out_crop, format="bgr24")


app = FastAPI()


@app.get("/")
async def index():
    return FileResponse('index.html')


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            t0 = time.time()

            rcv_bytes = await websocket.receive_bytes()
            print(f"bytes received {len(rcv_bytes)}")
            if len(rcv_bytes) <= 0:
                continue
            frame = av.VideoFrame.from_image(Image.open(io.BytesIO(rcv_bytes)))
            frame = hand_frame(frame)

            codec = av.CodecContext.create('vp9', 'w')
            codec.width = 512
            codec.height = 512
            codec.pix_fmt = 'yuv420p'
            codec.options = {'preset': 'fast'}

            _ = codec.encode(frame)
            packets = codec.encode(None)
            if len(packets) == 0:
                print("no packets, frame encode failed")
                continue

            for packet in packets:
                snd_bytes = bytes(packet)
                await websocket.send_bytes(snd_bytes)
                print(f"bytes sent {len(snd_bytes)}")

            time_taken = time.time() - t0
            print(f"time taken: {time_taken} ms")
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except EOFError:
            print("EOF")
        except Exception as e:
            print(f"{repr(e)}")


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
