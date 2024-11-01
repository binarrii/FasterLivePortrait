import os
import signal
import traceback
import uuid
import torch
import uvicorn
import requests

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from omegaconf import OmegaConf
from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline

conf_file = "configs/onnx_infer.yaml"
if torch.cuda.is_available():
    conf_file = "configs/trt_infer.yaml"

infer_cfg = OmegaConf.load(conf_file)
pipeline = GradioLivePortraitPipeline(cfg=infer_cfg)

tmp_path = "/tmp/livep"
os.makedirs(tmp_path, exist_ok=True)

app = FastAPI()


@app.post("/api/synthesis")
async def assets(request: Request):
    content_type = request.headers.get('Content-Type')
    if content_type.startswith('multipart/form-data'):
        req_form = await request.form()
    else:
        raise HTTPException(status_code=400, detail='Requires form data')
    
    text = req_form.get("text", None)

    portrait = req_form.get("portrait", None)
    if portrait:
        portrait_image_path = f"{tmp_path}/{portrait.filename}"
        with open(portrait_image_path, 'wb') as f:
            f.write(await portrait.read())
    else:
        raise HTTPException(status_code=400, detail='Portrait image is required')

    audio = req_form.get("audio", None)
    if audio:
        audio_file_path = f"{tmp_path}/{audio.filename}"
        with open(audio_file_path, 'wb+') as f:
            while True:
                chunk = await audio.read(1 << 13)
                if not chunk:
                    break
                f.write(chunk)

    motion = req_form.get("motion", None)
    if motion:
        motion_video_path = f"{tmp_path}/{motion.filename}"
        with open(motion_video_path, 'wb+') as f:
            while True:
                chunk = await motion.read(1 << 15)
                if not chunk:
                    break
                f.write(chunk)
    else:
        motion_video_path = None

    if not text and not audio:
        raise HTTPException(status_code=400, detail='Audio or text is required')

    url = 'http://192.168.1.73:9100/api/synthesis'
    files = {
        "portrait": (portrait.filename, open(portrait_image_path, 'rb'))
    }
    
    if audio:
        files["audio"] = (audio.filename, open(audio_file_path, 'rb'))
    if text:
        files["text"] = (None, text)

    with requests.post(url=url, files=files, stream=True) as response:
        if response.status_code == 200:
            driving_video_path = f"/tmp/{uuid.uuid4().hex}.mp4"
            with open(driving_video_path, 'wb+') as f:
                for chunk in response.iter_content(chunk_size=4096):
                    f.write(chunk)
        else:
            print(f"Error downloading file: {response.status_code}")
            return {"error": "synthesis failed"}

    output_video_path, _ = pipeline.execute_video(
        input_source_image_path=portrait_image_path,
        input_source_video_path=None,
        input_motion_video_path=motion_video_path,
        input_driving_video_path=driving_video_path,
        flag_relative_input=True,
        flag_do_crop_input=True,
        flag_remap_input=True,
        driving_multiplier=1.0,
        flag_stitching=True,
        flag_crop_driving_video_input=True,
        flag_video_editing_head_rotation=False,
        flag_is_animal=False,
        scale=3.2,
        vx_ratio=0.0,
        vy_ratio=0.0,
        scale_crop_driving_video=3.2,
        vx_ratio_crop_driving_video=0.0,
        vy_ratio_crop_driving_video=0.0,
        driving_smooth_observation_variance=1e-7,
        tab_selection='',
    )
    print(f"Final generated video path: {output_video_path}")

    def iterfile():
        with open(output_video_path, mode='rb') as f:
            yield from f

    return StreamingResponse(iterfile(), media_type='video/mp4')

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=9080, workers=1)
    except Exception as e:
        traceback.print_stack()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
