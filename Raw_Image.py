import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import datetime

# Define ROI parameters
ROI_CENTERS = [
    (2100, 1400) # Center of 4K frame
]
ROI_NAMES = ["ROI_1"]
ROI_WIDTH = 700   # Width of rectangle in pixels
ROI_HEIGHT = 350  # Height of rectangle in pixels

def setup_directories(test_id):
    """Set up all needed directories"""
    test_name = input("Enter test name: ")
    test_dir = f"{test_name}_test_{test_id}"
    
    subdirs = [
        "images/raw_captures",
        "images/roi"
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(test_dir, subdir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            
    return test_dir, test_name  # Return both values

def process_roi(raw_array, roi_center):
    """Process ROI using RAW data"""
    center_x, center_y = roi_center
    
    # Calculate rectangle boundaries
    x1 = max(0, center_x - ROI_WIDTH//2)
    y1 = max(0, center_y - ROI_HEIGHT//2)
    x2 = min(raw_array.shape[1], center_x + ROI_WIDTH//2)
    y2 = min(raw_array.shape[0], center_y + ROI_HEIGHT//2)
    
    return (x1, y1, x2-x1, y2-y1)



def main(test_id="001"):
    picam2 = None
    try:
        # Setup directories
        test_dir, test_name = setup_directories(test_id)
        
        # Initialize camera with RAW configuration
        picam2 = Picamera2()
        capture_config = picam2.create_still_configuration(
            raw={"size": picam2.sensor_resolution},  # RAW capture
            main={"size": (3840, 2160)},            # Display frame
            controls={"AfMode": 0}                  
        )
        preview_config = picam2.create_preview_configuration()
        picam2.configure(preview_config)
        
        # Add sharpness control
        lens_position = 3.5      # Initial lens position
        lens_step = 0.5        # Step size for focus adjustment
        picam2.set_controls({"LensPosition": lens_position})
        
        print("Starting RAW capture. Controls:")
        print("'q' - quit")
        print("'s' - save current frame")
        print("'+' - increase focus")
        print("'-' - decrease focus")
        
        # Initialize timing
        # interval = 2  # Capture every 2 seconds (commented out)
        # last_capture = 0  # (commented out)
        
        picam2.start()
        
        while True:
            current_time = time.time()
            
            # Capture RAW frame
            request = picam2.switch_mode_and_capture_request(capture_config)
            display_frame = request.make_array("main")
            
            # Process ROIs
            for i, center in enumerate(ROI_CENTERS):
                # Draw ROI on display
                cv2.rectangle(display_frame,
                            (center[0] - ROI_WIDTH//2, center[1] - ROI_HEIGHT//2),
                            (center[0] + ROI_WIDTH//2, center[1] + ROI_HEIGHT//2),
                            (255, 0, 0), 2)
                cv2.putText(display_frame, ROI_NAMES[i],
                          (center[0] - ROI_WIDTH//2, center[1] - ROI_HEIGHT//2 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # Comment out automatic capture
                # if current_time - last_capture >= interval:
                #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                #     roi_path = os.path.join(test_dir, "images/roi", 
                #                           f"{timestamp}_ROI_{ROI_NAMES[i]}.dng")
                #     request.save_dng(roi_path)
            
            # Comment out interval update
            # if current_time - last_capture >= interval:
            #     last_capture = current_time
            
            # Display
            display_frame = cv2.resize(display_frame, (1920, 1080))
            cv2.imshow("Main View", display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break  # This breaks the while loop
            elif key == ord('s'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Save full raw capture only
                raw_path = os.path.join(test_dir, "images/raw_captures", f"{timestamp}_full.dng")
                request.save_dng(raw_path)
                print(f"Saved full capture to: {raw_path}")
                

                
            elif key == ord('+') or key == ord('='):
                lens_position = min(100.00, lens_position + lens_step)
                picam2.set_controls({"LensPosition": lens_position})
                print(f"Lens position: {lens_position:.2f}")
            elif key == ord('-') or key == ord('_'):
                lens_position = max(0.0, lens_position - lens_step)
                picam2.set_controls({"LensPosition": lens_position})
                print(f"Lens position: {lens_position:.2f}")
            
            request.release()
            
        # Cleanup moved here, after the while loop
        if picam2:
            picam2.stop()
        cv2.destroyAllWindows()
        return  # Exit the function properly
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if picam2:
            picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_id = input("Enter test ID (e.g., 001): ")
    main(test_id)
    
    
