import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GLOG_v"] = "0"
os.environ["GLOG_logtostderr"] = "0"
os.environ["GLOG_minloglevel"] = "2"

import json
import logging

import threading
import time

import mediapipe as mp
import numpy as np
import torch
from omegaconf import OmegaConf
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions as mpBaseOptions

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av

av.logging.set_level(av.logging.FATAL)
av.logging.set_libav_level(av.logging.FATAL)

import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Portrait", layout="wide")

logger = logging.getLogger(__file__)
lock = threading.Lock()

face_detect_model = "checkpoints/liveportrait_onnx/face_detector.tflite"
conf_file = "configs/onnx_infer.yaml"
if torch.cuda.is_available():
    conf_file = "configs/trt_mp_infer.yaml"

infer_cfg = OmegaConf.load(conf_file)
infer_cfg.infer_params.flag_pasteback = False
pipe = FasterLivePortraitPipeline(cfg=infer_cfg)

mp_base_options = mpBaseOptions(model_asset_path=face_detect_model)
face_detect_options = vision.FaceDetectorOptions(
    base_options=mp_base_options,
    min_detection_confidence=0.5,
    min_suppression_threshold=0.3
)


def make_video_frame_callback():
    infer_times = []

    def frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
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
        with lock:
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
                    src_img = pipe.src_imgs[0].copy()
                    logger.info(f"src_o: w={src_img.shape[1]}, h={src_img.shape[0]}")
                    h, w = out_crop.shape[:2]
                    logger.info(f"out_o: w={w}, h={h}")
                    src_img = cv2.resize(src_img, (w, int(w / (w / h))))
                    logger.info(f"src_r: w={src_img.shape[1]}, h={src_img.shape[0]}")
                    src_img = src_img[h:src_img.shape[0], 0:w]
                    logger.info(f"src_c: w={src_img.shape[1]}, h={src_img.shape[0]}")
                    out_crop = cv2.vconcat([out_crop, src_img])
                    logger.info(f"out_n: w={out_crop.shape[1]}, h={out_crop.shape[0]}")
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

    return frame_callback


with open('ice.json') as f:
    COMMON_RTC_CONFIG = json.load(f)

col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.header("Driving Video")
    ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=COMMON_RTC_CONFIG,
        sendback_audio=False,
        media_stream_constraints={"video": True, "audio": True},
    )

with col_2:
    st.header("Input Image")
    file = st.file_uploader("SrcImage", type=["jpg", "png"], label_visibility="hidden")
    if file is not None:
        raw_bytes = file.read()
        with open(f"/tmp/{file.name}", "wb") as f:
            f.write(raw_bytes)
        np_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        cv_image = cv2.imdecode(np_bytes, 1)
        st.image(cv_image, channels="BGR")
        with lock:
            pipe.prepare_source(f"/tmp/{file.name}", realtime=True)

with col_3:
    st.header("Output Video")
    callback = make_video_frame_callback()
    webrtc_streamer(
        key="filter",
        mode=WebRtcMode.RECVONLY,
        video_frame_callback=callback,
        source_video_track=ctx.output_video_track,
        source_audio_track=ctx.output_audio_track,
        desired_playing_state=ctx.state.playing,
        rtc_configuration=COMMON_RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": True},
    )
