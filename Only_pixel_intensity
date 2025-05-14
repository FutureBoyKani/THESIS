import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import datetime
import csv

# Load calibration parameters from NPZ file
calibration_path = 'camera_calibration.npz'
calibration_data = np.load(calibration_path)
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Define your regions of interest (ROIs) as center points (x, y)
# Format: [(center_x, center_y), ...]
ROI_CENTERS = [
    # (400, 1550),     # ROI 1: center point
    # (1100, 1650),
    (1700, 1650)
    # (2200, 1650),
    # (2850, 1650),
    # (3500, 1700)
    
]

# ROI names
ROI_NAMES = ["0.1g/20ml", "0.2g/20ml", "0.4g/20ml", "0.6g/20ml", "0.8g/20ml", "1g/20ml"]

ROI_WIDTH = 40  # Width of rectangle in pixels
ROI_HEIGHT = 100  # Height of rectangle in pixels



def calculate_intensity_metrics(roi_image):
    """
    Calculate intensity metrics for an ROI
    
    Args:
        roi_image: Image of the ROI
        
    Returns:
        intensity_matrix: Matrix of pixel intensities
        avg_intensity: Average intensity of all pixels in the ROI
        std_intensity: Standard deviation of intensities
        min_intensity: Minimum intensity value
        max_intensity: Maximum intensity value
    """
    # Convert to grayscale if image is in color
    if len(roi_image.shape) == 3:
        gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi_image
    
    # Create a mask for non-zero pixels (part of the ROI)
    mask = np.where(gray_roi > 0, 1, 0).astype(np.uint8)
    
    # Extract intensity matrix (only for pixels in the ROI)
    intensity_matrix = gray_roi * mask
    
    # Calculate statistics (only for pixels in the ROI)
    non_zero_pixels = cv2.countNonZero(mask)
    if non_zero_pixels > 0:
        # Get only the non-zero values for statistics
        non_zero_values = intensity_matrix[intensity_matrix > 0]
        avg_intensity = np.mean(non_zero_values)
        std_intensity = np.std(non_zero_values)
        min_intensity = np.min(non_zero_values)
        max_intensity = np.max(non_zero_values)
    else:
        avg_intensity = std_intensity = min_intensity = max_intensity = 0
    
    return intensity_matrix, avg_intensity, std_intensity, min_intensity, max_intensity

def undistort_image(img):
    """
    Apply camera calibration to undistort an image
    
    Args:
        img: Image to undistort
        
    Returns:
        Undistorted image
    """
    if 'camera_matrix' not in calibration_data or 'dist_coeffs' not in calibration_data:    
        raise ValueError("Calibration file is missing 'camera_matrix' or 'dist_coeffs'")

    if img is None:
        raise ValueError("Input image is None. Make sure camera frame was captured correctly.")

    h, w = img.shape[:2]
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Convert the image to the expected format
    img_mat = cv2.UMat(img)
    
    # Undistort the image
    undistorted = cv2.undistort(img_mat, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Convert back to CPU array for further processing
    undistorted = undistorted.get()
    
    # Crop the image based on ROI from getOptimalNewCameraMatrix
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def process_roi(frame, roi_center):
    """Process a single rectangular region of interest"""
    center_x, center_y = roi_center
    
    # Calculate rectangle boundaries
    x1 = max(0, center_x - ROI_WIDTH//2)
    y1 = max(0, center_y - ROI_HEIGHT//2)
    x2 = min(frame.shape[1], center_x + ROI_WIDTH//2)
    y2 = min(frame.shape[0], center_y + ROI_HEIGHT//2)
    
    # Verify valid coordinates
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid ROI coordinates: ({x1}, {y1}, {x2}, {y2})")
    
    # Crop the frame to ROI
    cropped_frame = frame[y1:y2, x1:x2]
    
    # Create a mask for the rectangular ROI
    mask = np.ones(cropped_frame.shape[:2], dtype=np.uint8) * 255
    
    # Apply mask to get the ROI
    masked_roi = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
    
    # Calculate intensity metrics
    intensity_matrix, avg_intensity, std_intensity, min_intensity, max_intensity = calculate_intensity_metrics(masked_roi)
    
    return masked_roi, (x1, y1, x2-x1, y2-y1), intensity_matrix, avg_intensity, std_intensity, min_intensity, max_intensity

def setup_directories(test_id):
    """Set up all needed directories within a test-specific directory"""
    # Get test name from user
    test_name = input("Enter test name: ")
    
    # Create main test directory with test name
    test_dir = f"{test_name}_test_{test_id}"
    
    # Define subdirectories within the test directory
    subdirs = [
        "images/pi_timelapse",
        "images/roi/intensity",
        "data/csv",
        "data/intensity_matrix"
    ]
    
    # Create all directories
    for subdir in subdirs:
        full_path = os.path.join(test_dir, subdir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            
    return test_dir, test_name

# Resize function for image display and saving
def resize_quarter(image):
    """Resize an image to quarter of its original dimensions"""
    h, w = image.shape[:2]
    return cv2.resize(image, (w//4, h//4))

def visualize_intensity(intensity_matrix, window_name):
    """
    Create a visualization of pixel intensities
    
    Args:
        intensity_matrix: Matrix of pixel intensities
        window_name: Name for the visualization window
    
    Returns:
        heatmap: The generated heatmap visualization
    """
    # Normalize the intensity matrix to 0-255 range for visualization
    normalized = cv2.normalize(intensity_matrix, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to heatmap colorscheme
    heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Add text with min/max values
    min_val = np.min(intensity_matrix[intensity_matrix > 0]) if np.any(intensity_matrix > 0) else 0
    max_val = np.max(intensity_matrix)
    mean_val = np.mean(intensity_matrix[intensity_matrix > 0]) if np.any(intensity_matrix > 0) else 0
    
    # Add text to the image
    cv2.putText(heatmap, f"Min: {min_val:.1f}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(heatmap, f"Max: {max_val:.1f}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(heatmap, f"Mean: {mean_val:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display the heatmap
    display_heatmap = cv2.resize(heatmap, (heatmap.shape[1]//2, heatmap.shape[0]//2))
    cv2.imshow(window_name, display_heatmap)
    
    return heatmap

def main(test_id="001"):
    try:
        # Set up directories and get base test directory
        test_dir, test_name = setup_directories(test_id)
        
        # Update CSV file path for intensity data
        intensity_csv_file = os.path.join(test_dir, "data/csv", f"{test_name}_intensity_metrics.csv")
        
        # Setup intensity CSV
        with open(intensity_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp', 'elapsed_time']
            for name in ROI_NAMES:
                header.extend([f'{name}_avg_intensity', f'{name}_std_intensity', 
                             f'{name}_min_intensity', f'{name}_max_intensity'])
            writer.writerow(header)

        # Initialize camera
        picam2 = Picamera2()
        config = picam2.create_still_configuration(
            main={"size": (3840, 2160)},
            controls={"AfMode": 0}
        )
        
        picam2.configure(config)
        
        # Define capture intervals
        interval = 2                # Data capture interval in seconds
        last_capture_time = 0        # Timer for data capture
        last_screenshot_time = 0     # Timer for screenshot capture
        screenshot_interval = 900    # 15 minutes in seconds
        roi_interval = 60            # 1 minute for ROI intensity visualization 
        last_roi_time = 0 
        
        picam2.start()
        lens_position = 5.55        # Initial lens position for objects ~5cm away
        lens_step = 0.05            # Smaller step size for finer control
        picam2.set_controls({"LensPosition": lens_position})

        print("Starting intensity measurement. Controls:")
        print("'q' - quit")
        print("'s' - save current frame")
        print("'+' - increase focus")
        print("'-' - decrease focus")
        time.sleep(2)
        
        # Use the original ROI centers
        roi_centers = ROI_CENTERS
        
        start_time = time.time()
        
        while True:
            # Get current time at start of loop
            current_time = time.time()
            
            cal_frame = picam2.capture_array() # Normal Frame     

            # Uncomment these two lines when calibration is working
            # frame = picam2.capture_array()
            # cal_frame = undistort_image(frame) # Calibrated Frame
    
            display_frame = cal_frame.copy()
            
            # Process each ROI
            roi_results = []
            intensity_results = []
            
            for i, center in enumerate(roi_centers):
                center_x, center_y = center
                
                # Draw rectangular ROI on display frame (always blue, no selection color)
                cv2.rectangle(display_frame, 
                            (center_x - ROI_WIDTH//2, center_y - ROI_HEIGHT//2),
                            (center_x + ROI_WIDTH//2, center_y + ROI_HEIGHT//2),
                            (255, 0, 0), 2)  # Always blue color
                cv2.putText(display_frame, ROI_NAMES[i], 
                          (center_x - ROI_WIDTH//2, center_y - ROI_HEIGHT//2 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                          (255, 0, 0), 2)  # Always blue color
                
                # Process ROI and get intensity metrics
                masked_roi, roi_box, intensity_matrix, avg_intensity, std_intensity, min_intensity, max_intensity = process_roi(cal_frame, center)
                
                roi_results.append((masked_roi, roi_box))
                intensity_results.append((intensity_matrix, avg_intensity, std_intensity, min_intensity, max_intensity))

            # Resize display frame for smoother display
            display_frame = cv2.resize(display_frame, (1920, 1080))

            # Show main frame with ROI boxes
            cv2.imshow("Main View", display_frame)

            # Show windows for each ROI's masked area
            for i, (masked_roi, _) in enumerate(roi_results):
                # Resize ROI windows for display
                display_roi = cv2.resize(masked_roi, (masked_roi.shape[1]//2, masked_roi.shape[0]//2))
                
                # Show raw masked ROI
                cv2.imshow(f"ROI {i+1} - {ROI_NAMES[i]} Masked", display_roi)

            # Show and save intensity visualizations at specified interval
            if current_time - last_roi_time >= roi_interval:
                timestamp = datetime.datetime.now().strftime("%H_%M_%S")
                for i, (intensity_matrix, _, _, _, _) in enumerate(intensity_results):
                    # Create and show intensity visualization
                    heatmap = visualize_intensity(intensity_matrix, f"ROI {i+1} - {ROI_NAMES[i]} Intensity")
                    
                    # Save intensity visualization
                    intensity_path = os.path.join(test_dir, "images/roi/intensity", f"{timestamp}_ROI_{i+1}_intensity.jpg")
                    resized_intensity = resize_quarter(heatmap)
                    cv2.imwrite(intensity_path, resized_intensity)
                
                last_roi_time = current_time
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.datetime.now().strftime("%H_%M_%S")
                
                # Save main frame with quarter resolution
                main_img_path = os.path.join(test_dir, "images/pi_timelapse", f"{timestamp}_main.jpg")
                resized_frame = resize_quarter(cal_frame)
                cv2.imwrite(main_img_path, resized_frame)
                
                # Save each ROI's intensity visualization
                for i, (intensity_matrix, _, _, _, _) in enumerate(intensity_results):
                    heatmap = visualize_intensity(intensity_matrix, f"ROI {i+1} - {ROI_NAMES[i]} Intensity")
                    intensity_path = os.path.join(test_dir, "images/roi/intensity", f"{timestamp}_ROI_{i+1}_intensity.jpg")
                    resized_intensity = resize_quarter(heatmap)
                    cv2.imwrite(intensity_path, resized_intensity)
                
                print(f"Manually captured at {timestamp}")
                for i, (_, avg, std, min_val, max_val) in enumerate(intensity_results):
                    print(f"{ROI_NAMES[i]}: Avg: {avg:.2f} | Std: {std:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f}")
            
            # Focus control
            elif key == ord('+') or key == ord('='):
                lens_position = min(10.0, lens_position + lens_step)
                picam2.set_controls({"LensPosition": lens_position})
                print(f"Lens position: {lens_position:.2f}")
            elif key == ord('-') or key == ord('_'):
                lens_position = max(0.0, lens_position - lens_step)
                picam2.set_controls({"LensPosition": lens_position})
                print(f"Lens position: {lens_position:.2f}")
                    
            # Screenshot capture at defined interval
            if current_time - last_screenshot_time >= screenshot_interval:
                timestamp = datetime.datetime.now().strftime("%H_%M_%S")
                main_img_path = os.path.join(test_dir, "images/pi_timelapse", f"{timestamp}_main.jpg")
                resized_frame = resize_quarter(cal_frame)
                cv2.imwrite(main_img_path, resized_frame)
                last_screenshot_time = current_time
                print(f"Screenshots captured at {timestamp}")

            # Data capture for CSV at defined interval
            if current_time - last_capture_time >= interval:
                timestamp = datetime.datetime.now().strftime("%H_%M_%S")
                elapsed_time = current_time - start_time
                
                # Save intensity data
                intensity_row = [timestamp, f"{elapsed_time:.2f}"]
                for _, avg_int, std_int, min_int, max_int in intensity_results:
                    intensity_row.extend([avg_int, std_int, min_int, max_int])
                
                # Optional: Save intensity matrices to separate file
                # for i, (intensity_matrix, _, _, _, _) in enumerate(intensity_results):
                #     matrix_filename = os.path.join(
                #         test_dir,
                #         "data/intensity_matrix",
                #         f"{timestamp}_ROI_{i+1}_matrix.npy"
                #     )
                #     np.save(matrix_filename, intensity_matrix)
                
                # Write to CSV file
                with open(intensity_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(intensity_row)
                
                print(f"Data captured at {timestamp}")
                for i, (_, avg, std, min_val, max_val) in enumerate(intensity_results):
                    print(f"{ROI_NAMES[i]}: Avg: {avg:.2f} | Std: {std:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f}")
                
                last_capture_time = current_time

    except Exception as e:
        print(f"Error: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_id = input("Enter test ID (e.g., 001): ")
    main(test_id)
