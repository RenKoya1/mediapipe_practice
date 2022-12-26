import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 0, 255))
#
img_path = 'man.jpg'

with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    min_detection_confidence=0.5,
    static_image_mode=True) as face_mesh:
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize = None, fx = 0.3, fy = 0.3)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = rgb_img.shape[0]
    width = rgb_img.shape[1]

    results = face_mesh.process(rgb_img)

    annotated_img = img.copy()

    for face_landmarks in results.multi_face_landmarks:
        # for id, lm in enumerate(face_landmarks.landmark):
        #     print(id, lm.x)
        mp_drawing.draw_landmarks(
            image=annotated_img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mark_drawing_spec,
            connection_drawing_spec=mesh_drawing_spec
        )
        cv2.imwrite('result.jpg', annotated_img)