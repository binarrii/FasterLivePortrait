import cv2
from face_alignment import FaceAlignment, LandmarksType


class FaceAlignmentModel:

    def __init__(self, **kwargs):
        self.fa = FaceAlignment(LandmarksType.TWO_HALF_D, flip_input=False)

    def predict(self, *data):
        return self.fa.get_landmarks_from_image(cv2.cvtColor(data[0], cv2.COLOR_BGR2RGB))
