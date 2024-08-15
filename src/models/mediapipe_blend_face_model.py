# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeBlendFaceModel:

    def __init__(self, **kwargs):
        base_options = python.BaseOptions(model_asset_path=kwargs['model_path'])
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=False,
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def predict(self, *data):
        img_bgr = data[0]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        results = self.detector.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        ))

        # Print and draw face mesh landmarks on the image.
        if not results.face_landmarks:
            return []
        outs = []
        for face_landmarks in results.face_landmarks:
            landmarks = []
            for landmark in face_landmarks:
                # 提取每个关键点的 x, y, z 坐标
                landmarks.append([landmark.x * w, landmark.y * h])
            outs.append(np.array(landmarks))
        return outs
