import json
import time
import traceback

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

st.set_page_config(layout="wide")
st.markdown(
    """
Fork one input to multiple outputs with different video filters.
"""
)

infer_cfg = OmegaConf.load("configs/trt_mp_infer.yaml")
infer_cfg.infer_params.flag_pasteback = False
pipe = FasterLivePortraitPipeline(cfg=infer_cfg)


def make_video_frame_callback():
    infer_times = []

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        driving_frame = frame.to_ndarray(format="bgr24")

        t0 = time.time()
        try:
            dri_crop, out_crop, out_org = pipe.run(driving_frame, pipe.src_imgs[0], pipe.src_infos[0], realtime=True)
            infer_times.append(time.time() - t0)
            out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
            print(f"inference median time: {np.median(infer_times) * 1000} ms/frame, "
                  f"mean time: {np.mean(infer_times) * 1000} ms/frame")
        except:
            out_crop = driving_frame
            print(traceback.format_exc())

        return av.VideoFrame.from_ndarray(out_crop, format="bgr24")

    return callback


with open('ice.json') as f:
    iceServers = json.load(f)
    COMMON_RTC_CONFIG = {"iceServers": iceServers}

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
        img_src = pipe.prepare_source(f"/tmp/{file.name}", realtime=True)

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
