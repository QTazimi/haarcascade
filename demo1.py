import cv2
import numpy as np

# [x, y, w, h] = [0, 0, 0, 0]
#
# path = './face_images/2008_002506.jpg'
#
# face_Cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
#
# frame = cv2.imread(path)
#
# size = frame.shape[:2]
# image = np.zeros(size, dtype=np.float32)
# image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
# # 直方图均衡
# image = cv2.equalizeHist(image)
# im_h, im_w = size
# minSize_1 = (im_w // 10, im_h // 10)
# faceRects = face_Cascade.detectMultiScale(image, 1.05, 2, cv2.CASCADE_SCALE_IMAGE, minSize_1)
# if len(faceRects) > 0:
#     for faceRect in faceRects:
#         x, y, w, h = faceRect
#         cv2.rectangle(frame, (x, y), (x + w, y + h), [255, 255, 0], 2)
#
# cv2.imshow("detection", frame)
# cv2.waitKey(0)
print('okay')
