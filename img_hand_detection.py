import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 0, 255))
#

img_path = 'hand.jpg'

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    static_image_mode=True) as hands_detection:
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize = None, fx = 0.3, fy = 0.3)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = rgb_img.shape[0]
    width = rgb_img.shape[1]
    result = hands_detection.process(rgb_img)

    annotated_img = img.copy()

    for hand_landmarks in result.multi_hand_landmarks:
        # for id, lm in enumerate(hand_landmarks.landmark):
        #     print(id,lm.x)
        mp_drawing.draw_landmarks(
            image=annotated_img,
            landmark_list=hand_landmarks,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mark_drawing_spec,
            connection_drawing_spec=mesh_drawing_spec
        )
        cv2.imwrite('result.jpg', annotated_img)