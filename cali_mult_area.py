import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import datetime
import csv

PIXELS_PER_METER = 16436   # Calculate your pixels per meter

# Load calibration parameters from NPZ file
calibration_path = 'camera_calibration.npz'
calibration_data = np.load(calibration_path)
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Convert 4 cm diameter to pixel radius
CM_DIAMETER = 4
RADIUS_PIXELS = int((CM_DIAMETER / 2) * PIXELS_PER_METER / 100)  # Convert cm to meters, then to pixels

# Define your regions of interest (ROIs) as center points (x, y)
# Format: [(center_x, center_y), ...]
ROI_CENTERS = [
    (450, 500),      # ROI 1: center point
    (1850, 500)     # ROI 2: center point
    # (3350, 500),     # ROI 3: center point
    # (450, 1775),    # ROI 4: center point
    # (1850, 1775),     # ROI 5: center point
    # (3350, 1775)
]

# ROI names remain the same
ROI_NAMES = ["Shape 1", "Shape 2", "Shape 3", "Shape 4","Shape 5", "Shape 6"]

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
    
    # Undistort the image
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop the image based on ROI from getOptimalNewCameraMatrix
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def connect_edges_with_closing(edges, kernel_size=(5, 5)):
    """
    Connect edges from Canny edge detection using morphological closing.
    
    Args:
        edges: Binary edge image from Canny detector
        kernel_size: Size of the morphological kernel
        
    Returns:
        Processed image with connected edges
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed_edges

def connect_edges_multi_kernel(edges):
    # Start with small kernel, progressively increase
    result = edges.copy()
    for size in [3, 5, 7]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result

def canny(image, low_threshold=85, high_threshold=105, aperture_size=3):
    # Make a copy of the original image
    original = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detector
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)
    
    # Connect edges using closing algorithms
    connected_edges = connect_edges_multi_kernel(edges)
    return connected_edges

def area_calc(connected_edges, min_area=9912, max_area=5000000):
    """
    Calculate the total area of regions enclosed by connected edges.
    
    Args:
        connected_edges: Binary image with connected edges from Canny detection
        minimum_area: Minimum contour area to consider (filters out noise)
        
    Returns:
        total_area: Total area of all significant contours in pixels
        contours: List of significant contours found
    """
    # Create a copy of the connected edges
    edges_copy = connected_edges.copy()
    
    # # Area detection with only contours
    # picam2 = Picamera2()
    # config = picam2.create_still_configuration(
    # main={"size": (3840, 2160)},
    # controls={"AfMode": 1}
    # )

    # picam2.configure(config)
    # picam2.start()

    # # Give the camera a moment to adjust
    # time.sleep(1)
    # edges_copy = picam2.capture_array()
    
    # Find contours in the connected edges
    contours, _ = cv2.findContours(edges_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL
    
    significant_contours = []
    total_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            significant_contours.append(contour)
            total_area += area
            
    area_m2 = (total_area / (PIXELS_PER_METER ** 2))*10000  # Convert to cm²
    
    return total_area, significant_contours, area_m2

def process_roi(frame, roi_center):
    """Process a single circular region of interest"""
    center_x, center_y = roi_center
    
    # Create a mask for the circular ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)


    cv2.circle(mask, (center_x, center_y), RADIUS_PIXELS, 255, -1)
    
    # Extract the ROI using the mask
    roi_image = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Create a rectangular bounding box for the circle to work with
    x1 = max(0, center_x - RADIUS_PIXELS)
    y1 = max(0, center_y - RADIUS_PIXELS)
    x2 = min(frame.shape[1], center_x + RADIUS_PIXELS)
    y2 = min(frame.shape[0], center_y + RADIUS_PIXELS)
    
    # Crop to the bounding box
    cropped_roi = roi_image[y1:y2, x1:x2]
    
    # Apply edge detection to ROI
    edges = canny(cropped_roi)
    
    # Apply the mask to the edges to ensure we only process the circular area
    cropped_mask = mask[y1:y2, x1:x2]
    masked_edges = cv2.bitwise_and(edges, edges, mask=cropped_mask)
    
    # Calculate area in ROI
    pixel_area, contours, area_cm2 = area_calc(masked_edges)
    
    # Draw contours on ROI image
    contour_image = cropped_roi.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    return masked_edges, contour_image, pixel_area, contours, area_cm2, (x1, y1, x2-x1, y2-y1)

def setup_directories(test_id):
    """Set up all needed directories"""
    dirs = [
        f"pi_timelapse_images_test{test_id}",
        f"Area_CSV_file_test{test_id}",
        f"roi_images_test{test_id}"
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    return dirs

# Add this function to resize an image to half its size
def resize_half(image):
    """Resize an image to half its original dimensions"""
    h, w = image.shape[:2]
    return cv2.resize(image, (w//2, h//2))

def main(test_id="001"):
    try:
        # Set up directories
        dirs = setup_directories(test_id)
        
        # Initialize camera
        picam2 = Picamera2()
        config = picam2.create_still_configuration(
            main={"size": (3840, 2160)},
            controls={"AfMode": 0}
        )
        
        picam2.configure(config)
        interval = 10  # Interval in seconds
        last_capture_time = 0            # Add a second timer for screenshot capture
        last_screenshot_time = 0  
        screenshot_interval = 900 # 30 minutes in seconds
        roi_interval = 60
        last_roi_time = 0 
        
        picam2.start()
        picam2.set_controls({"LensPosition": 3})
        time.sleep(2)
        print("Starting edge detection. Press 'q' to quit, 's' to save current frame.")
        
        # Create CSV file with headers
        csv_file = os.path.join(f"Area_CSV_file_test{test_id}", f"Area_cm2_multi_roi_test{test_id}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp', 'elapsed_time']
            for name in ROI_NAMES:
                header.extend([f'{name}_pixel_area', f'{name}_area_cm2'])
            writer.writerow(header)

        # Use the original ROI centers directly
        roi_centers = ROI_CENTERS
        
        start_time = time.time()
        
        while True:
            cal_frame = picam2.capture_array() ##Normal Frame     

            # #these two lines are for the callibrated camera (NOT YET WORKING)
            # frame = picam2.capture_array()
            # cal_frame = undistort_image(frame) ##calibrated Frame


            display_frame = cal_frame.copy()
            
            # Process each ROI
            roi_results = []
            for i, center in enumerate(ROI_CENTERS):
                center_x, center_y = center
                
                # Draw ROI circle on display frame
                cv2.circle(display_frame, (center_x, center_y), RADIUS_PIXELS, (255, 0, 0), 2)
                cv2.putText(display_frame, ROI_NAMES[i], (center_x - RADIUS_PIXELS, center_y - RADIUS_PIXELS - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # Process this ROI
                edges, contour_image, pixel_area, contours, area_cm2, roi_box = process_roi(cal_frame, center)
                roi_results.append((edges, contour_image, pixel_area, contours, area_cm2, roi_box))
                
                # Display area info on frame
                area_text = f"{ROI_NAMES[i]}: {area_cm2:.2f} cm²"
                cv2.putText(display_frame, area_text, (center_x - RADIUS_PIXELS, center_y + RADIUS_PIXELS + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show main frame with ROI boxes
            cv2.imshow("Main View", display_frame)
            
            # Show separate windows for each ROI's edge detection and contours
            for i, (edges, contour_image, _, _, _, _) in enumerate(roi_results):
                cv2.imshow(f"ROI {i+1} - {ROI_NAMES[i]} Edges", edges)
                cv2.imshow(f"ROI {i+1} - {ROI_NAMES[i]} Contours", contour_image)
            
            current_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            
            # Manual capture with 's' key
            if key == ord('s'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save main frame with half resolution
                main_img_path = f"pi_timelapse_images_test{test_id}/main_{timestamp}.jpg"
                resized_frame = resize_half(cal_frame)
                cv2.imwrite(main_img_path, resized_frame)
                
                # Save each ROI with half resolution
                for i, (edges, contour_image, pixel_area, _, area_cm2, _) in enumerate(roi_results):
                    edge_path = f"roi_images_test{test_id}/{ROI_NAMES[i]}_edges_{timestamp}.jpg"
                    contour_path = f"roi_images_test{test_id}/{ROI_NAMES[i]}_contours_{timestamp}.jpg"
                    # Resize the images to half size
                    resized_edge = resize_half(edges)
                    resized_pic = resize_half(contour_image)
                    cv2.imwrite(edge_path, resized_edge)
                    cv2.imwrite(contour_path, resized_pic)
                
                print(f"Manually captured at {timestamp}")
                for i, (_, _, pixel_area, _, area_cm2, _) in enumerate(roi_results):
                    print(f"{ROI_NAMES[i]}: {pixel_area:.2f} px | {area_cm2:.2f} cm²")
                    
            # Screenshot capture every 30 minutes
            if current_time - last_screenshot_time >= screenshot_interval:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Save main frame
                main_img_path = f"pi_timelapse_images_test{test_id}/main_{timestamp}.jpg"
                resized_frame = resize_half(cal_frame)
                cv2.imwrite(main_img_path, resized_frame)
                last_screenshot_time = current_time
                print(f"Screenshots captured at {timestamp}")

            # Take screenshots of ROI
            if current_time - last_roi_time >= roi_interval:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Save each ROI
                for i, (edges, contour_image, pixel_area, _, area_cm2, _) in enumerate(roi_results):
                    edge_path = f"roi_images_test{test_id}/{ROI_NAMES[i]}_edges_{timestamp}.jpg"
                    contour_path = f"roi_images_test{test_id}/{ROI_NAMES[i]}_contours_{timestamp}.jpg"
                    # Resize the taken images from camera
                    resized_edge = resize_half(edges)
                    resized_pic = resize_half(contour_image)
                    cv2.imwrite(edge_path, resized_edge)
                    cv2.imwrite(contour_path, resized_pic)
                last_roi_time = current_time
                
                # Also capture data for CSV when taking ROI screenshots
                if current_time - last_capture_time >= interval:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    elapsed_time = current_time - start_time
                    
                    # Save ROI data
                    row_data = [timestamp, f"{elapsed_time:.2f}"]
                    
                    # Add each ROI's data to CSV row
                    for i, (_, _, pixel_area, _, area_cm2, _) in enumerate(roi_results):
                        row_data.extend([pixel_area, area_cm2])
                        print(f"{ROI_NAMES[i]}: {pixel_area:.2f} px | {area_cm2:.2f} cm²")
                    
                    # Write to CSV
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row_data)
                    
                    last_capture_time = current_time

            # Data capture for CSV every 10 seconds
            if current_time - last_capture_time >= interval:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                elapsed_time = current_time - start_time
                
                # Save ROI data
                row_data = [timestamp, f"{elapsed_time:.2f}"]
                
                # Add each ROI's data to CSV row
                for i, (_, _, pixel_area, _, area_cm2, _) in enumerate(roi_results):
                    row_data.extend([pixel_area, area_cm2])
                    print(f"{ROI_NAMES[i]}: {pixel_area:.2f} px | {area_cm2:.2f} cm²")
                
                # Write to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                
                last_capture_time = current_time
            
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_id = input("Enter test ID (e.g., 001): ")
    main(test_id)
