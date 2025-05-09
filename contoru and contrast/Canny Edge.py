import cv2
import numpy as np
from picamera2 import Picamera2
import time 
from time import sleep
import os
import datetime

def canny_edge_detector(image, low_threshold=10, high_threshold=15, aperture_size=3): 
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:  # Fixed condition check
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5,5), 0)   
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)
    return edges

# Create directory for storing images if it doesn't exist
output_dir = "pi_timelapse_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    try:
        # Initialize Picamera2
        picam2 = Picamera2()
        config = picam2.create_still_configuration(
            main={"size": (1920, 1080)},
            controls={"AfMode": 2, "AfSpeed": 1}  # 2 = Continuous AF, 1 = Fast AF speed
        )

        
        
        # Configure camera
        config = picam2.create_still_configuration(main={"size": (1920, 1080)})
        picam2.configure(config)  # Fixed typo
        x = 5  # interval time in seconds
        last_capture_time = 0  # Track when the last image was saved
        
        # Start the camera
        picam2.start()
        
        # Allow the camera to warm up
        time.sleep(2)
        
        print("Starting edge detection. Press 'q' to quit.")
        
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Perform edge detection
            edges = canny_edge_detector(frame)
            
            # Display original and edge-detected images
            cv2.imshow("Original", frame)
            cv2.imshow("Canny Edges", edges)
            
            # Check if it's time to save an image
            current_time = time.time()
            if current_time - last_capture_time >= x:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                edge_pic = f"{output_dir}/edge_image_{timestamp}.jpg"
                
                
                # Save both edge detection and original images
                cv2.imwrite(edge_pic, edges)
                
                
                print(f"Captured: {edge_pic}")
                last_capture_time = current_time
            
            # Wait for 'q' key to exit (with a short delay to not hog CPU)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Stop the camera and close windows
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    