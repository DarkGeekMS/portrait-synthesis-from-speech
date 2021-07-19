from change_pose import FaceRotator
import cv2

renderer = FaceRotator()
img1 = cv2.imread("face_data/Images/target.jpg")
rotated = renderer.rotate_face(img1, 30)
rotated = renderer.rotate_face(img1, 60)
rotated = renderer.rotate_face(img1, 90)
rotated = renderer.rotate_face(img1, -30)
rotated = renderer.rotate_face(img1, -60)
rotated = renderer.rotate_face(img1, -90)
