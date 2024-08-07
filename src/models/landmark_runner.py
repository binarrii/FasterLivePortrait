# coding: utf-8

import cv2
import numpy as np

from .base_model import BaseModel
from .timer import Timer
from .crop import crop_image, _transform_pts


def to_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


class LandmarkRunner(BaseModel):
    """landmark runner"""

    def __init__(self, **kwargs):
        # super(LandmarkRunner, self).__init__(**kwargs)
        ckpt_path = kwargs.get('model_path')
        onnx_provider = kwargs.get('onnx_provider', 'cuda')
        device_id = kwargs.get('device_id', 0)
        self.dsize = kwargs.get('dsize', 224)
        self.timer = Timer()
        self.predictor = None

        import onnxruntime

        if onnx_provider.lower() == 'cuda':
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=[
                    ('CUDAExecutionProvider', {'device_id': device_id})
                ]
            )
        else:
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 4
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=['CPUExecutionProvider'],
                sess_options=opts
            )

        from media_pipe import LMKExtractor
        self.lmk_extractor = LMKExtractor()

    def input_process(self, *data):
        pass

    def output_process(self, *data):
        pass

    def predict(self, *data):
        img_rgb = data[0]
        face_result = self.lmk_extractor(img_rgb)
        face_index = 0

        if face_result is None:
            ret_dct = {}
            cropped_image_256 = None
            return ret_dct, cropped_image_256

        face_landmarks = face_result[face_index]

        lmks = []
        for index in range(len(face_landmarks)):
            x = face_landmarks[index].x * img_rgb.shape[1]
            y = face_landmarks[index].y * img_rgb.shape[0]
            lmks.append([x, y])
        pts = np.array(lmks)

        # crop the face
        ret_dct, image_crop = crop_image(
            img_rgb,  # ndarray
            pts,  # 106x2 or Nx2
            # dsize=dsize,
            # scale=scale,
            # vy_ratio=vy_ratio,
            # vx_ratio=vx_ratio,
            # rotate=rotate
        )
        # update a 256x256 version for network input or else
        cropped_image_256 = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_AREA)
        ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / 224

        input_image_size = img_rgb.shape[:2]
        ret_dct['input_image_size'] = input_image_size

        return self.run(img_rgb, pts)['pts']

    def _run(self, inp):
        out = self.session.run(None, {'input': inp})
        return out

    def run(self, img_rgb: np.ndarray, lmk=None):
        if lmk is not None:
            crop_dct, img_crop_rgb = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
        else:
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_lst = self._run(inp)
        out_pts = out_lst[2]

        pts = to_ndarray(out_pts[0]).reshape(-1, 2) * self.dsize  # scale to 0-224
        pts = _transform_pts(pts, M=crop_dct['M_c2o'])
        del crop_dct, img_crop_rgb
        return {
            'pts': pts,  # 2d landmarks 203 points
        }

    def warmup(self):
        self.timer.tic()

        dummy_image = np.zeros((1, 3, self.dsize, self.dsize), dtype=np.float32)

        _ = self._run(dummy_image)

        elapse = self.timer.toc()
        print(f'LandmarkRunner warmup time: {elapse:.3f}s')

    def __del__(self):
        pass
