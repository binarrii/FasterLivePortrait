import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# 定义嘴巴和眼睛的关键点索引
MOUTH_LANDMARKS     = [13, 14]  # 上下唇的关键点
LEFT_EYE_LANDMARKS  = [159, 145]  # 左眼上下眼睑的关键点
RIGHT_EYE_LANDMARKS = [386, 374]  # 右眼上下眼睑的关键点


LEFT_EYE  = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
RIGHT_EYE = [ 33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

prev_left_pupil  = None
prev_right_pupil = None

def calculate_distance(landmark1, landmark2, image_shape):
    ih, iw, _ = image_shape
    x1, y1 = int(landmark1.x * iw), int(landmark1.y * ih)
    x2, y2 = int(landmark2.x * iw), int(landmark2.y * ih)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

cap = cv2.VideoCapture(0)  # 使用摄像头
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    img_h, img_w = frame.shape[:2]

    # 转换图像颜色
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if len(face_landmarks.landmark) >= 478:
                # print(f"points: {len(face_landmarks.landmark)}")

                # 计算嘴巴张开幅度
                mouth_opening     = calculate_distance(face_landmarks.landmark[MOUTH_LANDMARKS[0]],     face_landmarks.landmark[MOUTH_LANDMARKS[1]], frame.shape)
                # 计算左眼睁开幅度
                left_eye_opening  = calculate_distance(face_landmarks.landmark[LEFT_EYE_LANDMARKS[0]],  face_landmarks.landmark[LEFT_EYE_LANDMARKS[1]], frame.shape)
                # 计算右眼睁开幅度
                right_eye_opening = calculate_distance(face_landmarks.landmark[RIGHT_EYE_LANDMARKS[0]], face_landmarks.landmark[RIGHT_EYE_LANDMARKS[1]], frame.shape)

                n = 0

                # 在图像上显示结果
                cv2.putText(frame, f'Mouth: {mouth_opening:.3f}', (10, (n:=n+1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Left  Eye: {left_eye_opening:.3f}',  (10, (n:=n+1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Right Eye: {right_eye_opening:.3f}', (10, (n:=n+1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 标记瞳孔位置
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in face_landmarks.landmark])
                cv2.polylines(frame, [mesh_points[LEFT_IRIS]],  True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0, 255, 0), 1, cv2.LINE_AA)

                left_pupil  = np.array([face_landmarks.landmark[468].x, face_landmarks.landmark[468].y])
                right_pupil = np.array([face_landmarks.landmark[473].x, face_landmarks.landmark[473].y])

                # 计算瞳孔移动值
                if prev_left_pupil is not None and prev_right_pupil is not None:
                    left_movement  = np.linalg.norm(left_pupil - prev_left_pupil)
                    right_movement = np.linalg.norm(right_pupil - prev_right_pupil)
                    cv2.putText(frame, f'Left  Pupil: [{left_pupil[0]:.3f}, {left_pupil[1]:.3f}]',  (10, (n:=n+1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f'Right Pupil: [{right_pupil[0]:.3f}, {right_pupil[1]:.3f}]', (10, (n:=n+1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f'Left  Pupil Movement: {left_movement:.3f}',  (10, (n:=n+1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f'Right Pupil Movement: {right_movement:.3f}', (10, (n:=n+1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # print(f"Left Pupil Movement: {left_movement}, Right Pupil Movement: {right_movement}")

                prev_left_pupil  = left_pupil
                prev_right_pupil = right_pupil

    cv2.imshow('FaceMesh', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

