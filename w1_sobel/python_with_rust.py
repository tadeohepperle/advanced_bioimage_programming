# python python_with_rust.py img/lizard.jpg

import subprocess
import sys


def generate_sobel_image(image_path):
    subprocess.run(["w1_sobel.exe", image_path])


if __name__ == "__main__":
    for image_path in sys.argv[1:]:
        generate_sobel_image(image_path)
