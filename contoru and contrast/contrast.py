import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import datetime
import csv


def black_and_white(image, threshold_value=100, max_value=255):
    """
    Convert the image to black and white using adjustable thresholds
    
    Args:
        image: Input image
        threshold_value: Pixels below this value become black (0-255)
        max_value: Value for pixels above threshold (0-255)
    
    Returns:
        Binary image where pixels are either 0 or max_value
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
    return bw_image

def count_black_pixels(binary_image):
    """
    Count the number of black pixels (0 values) in a binary image
    
    Args:
        binary_image: Binary image where pixels are either 0 or max_value
    
    Returns:
        num_black: Number of black pixels
        percentage: Percentage of black pixels in the image
    """
    # Count black pixels (where pixel value is 0)
    num_black = cv2.countNonZero(cv2.bitwise_not(binary_image))
    
    # Calculate total pixels and percentage
    total_pixels = binary_image.shape[0] * binary_image.shape[1]
    percentage = (num_black / total_pixels) * 100
    
    return num_black, percentage

def main():
    # Read image
    try:
        image = cv2.imread("Pixel_intensity_beads_only_test_007/images/pi_timelapse/14_00_00_main.jpg")
        if image is None:
            raise FileNotFoundError("Could not load image")
        
        # Show original image
        cv2.imshow("Original", image)
        
        # Convert the image to black and white with different thresholds
        bw_image_default = black_and_white(image)  # default threshold=100
        bw_image_dark = black_and_white(image)  # more black pixels

        # Count black pixels in both versions
        default_black, default_percent = count_black_pixels(bw_image_default)
        dark_black, dark_percent = count_black_pixels(bw_image_dark)

        # Print results
        print(f"\nPixel Analysis Results:")
        print(f"Default threshold:")
        print(f"- Black pixels: {default_black:,}")
        print(f"- Percentage: {default_percent:.2f}%")
        print(f"\nDark threshold:")
        print(f"- Black pixels: {dark_black:,}")
        print(f"- Percentage: {dark_percent:.2f}%")

        # Show all versions with pixel counts in window title
        cv2.imshow(f"Default Threshold (Black: {default_percent:.1f}%)", bw_image_default)
        cv2.imshow(f"Dark Threshold (Black: {dark_percent:.1f}%)", bw_image_dark)
        
        # Save the images
        cv2.imwrite("original_image.jpg", image)
        cv2.imwrite("bw_dark.jpg", bw_image_dark)
        
        # Wait for key press and close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

