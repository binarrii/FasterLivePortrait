import json
import time

import numpy as np
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

st.markdown(
    """
Fork one input to multiple outputs with different video filters.
"""
)


def make_video_frame_callback():
    infer_cfg = OmegaConf.load("configs/onnx_infer.yaml")
    infer_cfg.infer_params.flag_pasteback = False
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg)
    img_src = pipe.prepare_src_image("assets/examples/source/s7.jpg", realtime=True)
    infer_times = []

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        driving_frame = frame.to_ndarray(format="bgr24")

        t0 = time.time()
        try:
            dri_crop, out_crop, out_org = pipe.run(driving_frame, img_src)
            infer_times.append(time.time() - t0)
            out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        except:
            out_crop = driving_frame

        print(f"inference median time: {np.median(infer_times) * 1000} ms/frame, "
              f"mean time: {np.mean(infer_times) * 1000} ms/frame")

        return av.VideoFrame.from_ndarray(out_crop, format="bgr24")

    return callback


with open('ice.json') as f:
    iceServers = json.load(f)
    COMMON_RTC_CONFIG = {"iceServers": iceServers}

col_1, col_2 = st.columns(2)

with col_1:
    st.header("Input")
    ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=COMMON_RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

with col_2:
    st.header("Output")
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
