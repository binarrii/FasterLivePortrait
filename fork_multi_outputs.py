import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GLOG_v"] = "0"

import json
import logging

import threading
import time

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(layout="wide")

logger = logging.getLogger(__file__)
lock = threading.Lock()

face_detect_model = "checkpoints/liveportrait_onnx/face_detector.tflite"
conf_file = "configs/onnx_infer.yaml"
if torch.cuda.is_available():
    conf_file = "configs/trt_mp_infer.yaml"

infer_cfg = OmegaConf.load(conf_file)
infer_cfg.infer_params.flag_pasteback = False
pipe = FasterLivePortraitPipeline(cfg=infer_cfg)


def make_video_frame_callback():
    infer_times = []
    fail_times = [0]

    # prev_crop = [None]

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
                    if len(infer_times) % 10 == 0:
                        driving_frame_rgb = cv2.cvtColor(driving_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=driving_frame_rgb)
                        base_options = python.BaseOptions(model_asset_path=face_detect_model)
                        options = vision.FaceDetectorOptions(base_options=base_options,
                                                             min_detection_confidence=0.35,
                                                             min_suppression_threshold=0.25)
                        detector = vision.FaceDetector.create_from_options(options)
                        faces = detector.detect(mp_image)
                        if faces is not None and len(faces.detections) > 0:
                            # prev_crop[0] = out_crop
                            fail_times[0] = 0
                        else:
                            # prev_crop[0] = None
                            fail_times[0] = fail_times[0] + 1
                            if fail_times[0] > 1:
                                pipe.src_lmk_pre = None
                                fail_times[0] = 1
                                raise Exception("No face detected")
                finally:
                    if len(infer_times) > 7200:
                        infer_times.pop(0)
                    infer_times.append(time.time() - t0)

                print(f"inference median time: {np.median(infer_times) * 1000} ms/frame, "
                      f"mean time: {np.mean(infer_times) * 1000} ms/frame")
            except Exception as e:
                if len(infer_times) % 60 == 0:
                    logging.warning(f"{repr(e)}")
                out_crop = driving_frame
                # if prev_crop[0] is not None:
                #     out_crop = prev_crop[0]
                # else:
                #     src_img = cv2.cvtColor(pipe.src_imgs[0], cv2.COLOR_BGR2RGB) if len(pipe.src_imgs) > 0 else None
                #     out_crop = src_img if src_img is not None else driving_frame

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
        media_stream_constraints={"video": True, "audio": False},
    )

with col_2:
    st.header("Input Image")
    file = st.file_uploader("Upload a Image", type=["jpg", "png"])
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
        source_video_track=ctx.input_video_track,
        desired_playing_state=ctx.state.playing,
        rtc_configuration=COMMON_RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )
