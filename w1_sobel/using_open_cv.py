# pip install numpy
# pip install opencv-python
import cv2
import time
import numpy as np

img = cv2.imread("img/lizard.jpg")



img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)  # x
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)  # y

start = time.time()
print("hello")

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
sobel_img = gx + gy

end = time.time()
print(f"total time: {end - start}")
cv2.imwrite("img/lizard_cv.jpg", sobel_img)
