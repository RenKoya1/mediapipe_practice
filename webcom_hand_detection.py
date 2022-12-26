import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 0, 255))
#

cap_file = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    static_image_mode=False) as hands_detection:

    while cap_file.isOpened():
        ret, frame = cap_file.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands_detection.process(rgb_img)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mark_drawing_spec,
                    connection_drawing_spec=mesh_drawing_spec
                )
        cv2.imshow('hand detection', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap_file.release()    