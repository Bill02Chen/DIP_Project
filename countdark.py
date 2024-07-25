import cv2
import numpy as np


def count_dark_pixels(image, threshold=25):
    # Read the image
    # image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Count the number of dark pixels
    dark_pixels = gray_image < threshold
    # Count the number of dark pixels
    dark_pixels_count = np.count_nonzero(dark_pixels)
    total_pixels = gray_image.size
    percentage_dark = (dark_pixels_count / total_pixels) * 100

    # print(f'Total number of pixels: {total_pixels}')
    # print(f'Number of dark pixels (below threshold {threshold}): {dark_pixels_count}')
    # print(f'Percentage of dark pixels: {percentage_dark:.2f}%')

    return percentage_dark > 50

# # Example usage:
# image_path = '/Users/bill/Desktop/test/group.jpg'
# count_dark_pixels(image_path)