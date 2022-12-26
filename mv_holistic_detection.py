import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 0, 255))
#

cap_file = cv2.VideoCapture('dance.mp4')

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    static_image_mode=False) as holistic_detection: #false because movie
    while cap_file.isOpened():
        ret, frame = cap_file.read()
        if not ret:
            break

        frame = cv2.resize(frame, dsize = None, fx = 0.3, fy = 0.3)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic_detection.process(rgb_img)
        mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.pose_landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mesh_drawing_spec
        )
        mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.face_landmarks,
        connections=mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
        )
        mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
        )
        mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
        )
        cv2.imshow('holistic detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap_file.release()

        
