# python index.py img/lizard.jpg

import subprocess
import sys


def generate_sobel_image(image_path):
    subprocess.run(["target/release/w1_sobel.exe", image_path])


if __name__ == "__main__":
    for image_path in sys.argv[1:]:
        generate_sobel_image(image_path)
