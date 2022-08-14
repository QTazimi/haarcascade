import cv2
import numpy as np

[x, y, w, h] = [0, 0, 0, 0]

video_capture = cv2.VideoCapture("./video/hamilton_clip.mp4")
# video_capture = cv2.VideoCapture(0)

face_Cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")

cv2.namedWindow("Face Detection System")

while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    size = frame.shape[:2]
    image = np.zeros(size, dtype=np.float32)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    image = cv2.equalizeHist(image)
    im_h, im_w = size
    minSize_1 = (im_w // 8, im_h // 8)
    faceRects = face_Cascade.detectMultiScale(image, 1.05, 2, cv2.CASCADE_SCALE_IMAGE, minSize_1)
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 255, 0], 2)

    cv2.imshow("Face Detection System", frame)
    key = cv2.waitKey(5)
    if key == int(30):
        break

video_capture.release()
cv2.destroyWindow("Face Detection System")