# pip install numpy
# pip install opencv-python

# python using_open_cv.py img/lizard.jpg img/house.jpg img/ball.jpg

import sys
import cv2
import time
import numpy as np


if __name__ == "__main__":
    for image_path in sys.argv[1:]:
        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        start = time.time()

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        sobel_img = np.sqrt(np.multiply(gy, gy) +
                            np.multiply(gx, gx))

        end = time.time()
        print(f"{image_path} with size {img.shape[0]}x{img.shape[1]}")
        print(f"    Sobel done: {(end - start)*1000}ms")
        cv2.imwrite(image_path.split(".")[
                    0] + "_sobel_open_cv_native.jpg", sobel_img)
