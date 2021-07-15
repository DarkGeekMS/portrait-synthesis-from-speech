from change_pose import FaceRotator
import cv2

renderer = FaceRotator()
img = cv2.imread("face_data/Images/target.jpg")
rotated =renderer.rotate_face(img,0)
rotated = renderer.rotate_face(img, 30, reuse=True)
rotated = renderer.rotate_face(img, 60, reuse=True)
rotated = renderer.rotate_face(img,90, reuse=True)
