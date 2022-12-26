import mediapipe as mp
import cv2

mp_face_detect = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

kp_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))
bbox_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 0, 255))
#

img_path = 'man.jpg'

with mp_face_detect.FaceDetection(min_detection_confidence=0.5) as face_detection:
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize = None, fx = 0.3, fy = 0.3)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = rgb_img.shape[0]
    width = rgb_img.shape[1]

    result = face_detection.process(rgb_img)

    annotated_img = img.copy()

    for detection in result.detections:
        nose_x = mp_face_detect.get_key_point(detection, mp_face_detect.FaceKeyPoint.NOSE_TIP).x
        nose_y = mp_face_detect.get_key_point(detection, mp_face_detect.FaceKeyPoint.NOSE_TIP).y
        
        re_nose_x = int(nose_x*width)
        re_nose_y = int(nose_y*height)
        
        mp_drawing.draw_detection(annotated_img, detection, keypoint_drawing_spec=kp_drawing_spec, 
                                    bbox_drawing_spec=bbox_drawing_spec)

        cv2.imwrite('result.jpg', annotated_img)



    