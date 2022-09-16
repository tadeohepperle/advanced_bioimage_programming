import sys
import cv2
import time
import numpy as np


# python python_manual.py img/lizard.jpg img/house.jpg img/ball.jpg

def convert_to_greyscale(img):
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    grey_channel = 0.2 * blue + 0.5 * green + 0.3 * red
    return grey_channel


sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def convolution(img, kernel_3x3):
    w, h = img.shape
    stacked_sub_images = np.zeros((w-2, h-2))
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            kv = kernel_3x3[i, j]
            subimg = img[0+i:w-2+i, 0+j:h-2+j] * kv
            stacked_sub_images += subimg
    new_img = np.zeros((w, h))
    new_img[1:-1, 1:-1] = stacked_sub_images
    return new_img


if __name__ == "__main__":
    for image_path in sys.argv[1:]:
        img = cv2.imread(image_path)  # bgr
        grey_img = convert_to_greyscale(img)

        start = time.time()

        sobel_img_x = convolution(grey_img, sobel_kernel_x)
        sobel_img_y = convolution(grey_img, sobel_kernel_y)
        sobel_img = np.sqrt(np.multiply(sobel_img_y, sobel_img_y) +
                        np.multiply(sobel_img_x, sobel_img_x))

        end = time.time()
        print(f"total time: {end - start}")

        cv2.imwrite(image_path.split(".")[0] + "_sobel_python.jpg", sobel_img)



