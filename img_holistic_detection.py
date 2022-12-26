import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 0, 255))

img_path = 'man1.jpg'

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    static_image_mode=True) as holistic_detection:
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize = None, fx = 0.3, fy = 0.3)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = rgb_img.shape[0]
    width = rgb_img.shape[1]

    results = holistic_detection.process(rgb_img)

    annotated_img = img.copy()

    mp_drawing.draw_landmarks(
        image=annotated_img,
        landmark_list=results.pose_landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
    )
    mp_drawing.draw_landmarks(
        image=annotated_img,
        landmark_list=results.face_landmarks,
        connections=mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
    )
    mp_drawing.draw_landmarks(
        image=annotated_img,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
    )
    mp_drawing.draw_landmarks(
        image=annotated_img,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
    )
    cv2.imwrite('result.jpg', annotated_img)