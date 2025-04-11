import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import datetime
import csv

PIXELS_PER_METER = 15748   # Calculate your pixels per meter

# Define your regions of interest (ROIs)
# Format: [(x_start, y_start, width, height), ...]
# Example with 2 regions:
ROIS = [
    (1000, 800, 700, 600),    # ROI 1: (x, y, width, height)
    (2000, 800, 700, 600)    # ROI 2: (x, y, width, height)
]

# You can add more ROIs as needed
ROI_NAMES = ["Plastic_1_in_water", "Plastic_2_in_acid"]  # Names for each ROI

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

def canny(image, low_threshold=15, high_threshold=27, aperture_size=3):
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

def area_calc(connected_edges, min_area=9912, max_area=10000000):
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
    
    # Find contours in the connected edges
    contours, _ = cv2.findContours(edges_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Filter contours by minimum area and calculate total area
    # significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area]
    # largest_contour = max(significant_contours, key=cv2.contourArea)
    # total_area = cv2.contourArea(largest_contour)
    
    significant_contours = []
    total_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            significant_contours.append(contour)
            total_area += area
            
    area_m2 = (total_area / (PIXELS_PER_METER ** 2))*10000  # Convert to cm²
    
    return total_area, significant_contours, area_m2

def process_roi(frame, roi):
    """Process a single region of interest"""
    x, y, w, h = roi
    roi_image = frame[y:y+h, x:x+w]
    
    # Apply edge detection to ROI
    edges = canny(roi_image)
    
    # Calculate area in ROI
    pixel_area, contours, area_cm2 = area_calc(edges)
    
    # Draw contours on ROI image
    contour_image = roi_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    return edges, contour_image, pixel_area, contours, area_cm2

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
    h, w = image.shape[:4]
    return cv2.resize(image, (w//4, h//4))

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
        screenshot_interval = 1800  # 30 minutes in seconds
        roi_interval = 120
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
        
        start_time = time.time()
        
        while True:
            frame = picam2.capture_array()
            display_frame = frame.copy()
            
            # Process each ROI
            roi_results = []
            for i, roi in enumerate(ROIS):
                x, y, w, h = roi
                
                # Draw ROI rectangle on display frame
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(display_frame, ROI_NAMES[i], (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # Process this ROI
                edges, contour_image, pixel_area, contours, area_cm2 = process_roi(frame, roi)
                roi_results.append((edges, contour_image, pixel_area, contours, area_cm2))
                
                # Display area info on frame
                area_text = f"{ROI_NAMES[i]}: {area_cm2:.2f} cm²"
                cv2.putText(display_frame, area_text, (x, y+h+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show main frame with ROI boxes
            cv2.imshow("Main View", display_frame)
            
            # Show separate windows for each ROI's edge detection and contours
            for i, (edges, contour_image, _, _, _) in enumerate(roi_results):
                cv2.imshow(f"ROI {i+1} - {ROI_NAMES[i]} Edges", edges)
                cv2.imshow(f"ROI {i+1} - {ROI_NAMES[i]} Contours", contour_image)
            
            current_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            
            # # Manual capture with 's' key
            # if key == ord('s'):
            #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
            #     # Save main frame
            #     main_img_path = f"pi_timelapse_images_test{test_id}/main_{timestamp}.jpg"
            #     cv2.imwrite(main_img_path, frame)
                
            #     # Save each ROI
            #     for i, (edges, contour_image, pixel_area, _, area_cm2) in enumerate(roi_results):
            #         edge_path = f"roi_images_test{test_id}/{ROI_NAMES[i]}_edges_{timestamp}.jpg"
            #         contour_path = f"roi_images_test{test_id}/{ROI_NAMES[i]}_contours_{timestamp}.jpg"
            #         cv2.imwrite(edge_path, edges)
            #         cv2.imwrite(contour_path, contour_image)
                
            #     print(f"Manually captured at {timestamp}")
            #     for i, (_, _, pixel_area, _, area_cm2) in enumerate(roi_results):
            #         print(f"{ROI_NAMES[i]}: {pixel_area:.2f} px | {area_cm2:.2f} cm²")
            

            # In your main loop:
            current_time = time.time()

            # Screenshot capture every 30 minutes
            if current_time - last_screenshot_time >= screenshot_interval:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save main frame
                main_img_path = f"pi_timelapse_images_test{test_id}/main_{timestamp}.jpg"
                resized_frame = resize_half(frame)
                cv2.imwrite(main_img_path, resized_frame)              
                last_screenshot_time = current_time
                print(f"Screenshots captured at {timestamp}")
             
            #take a screenshots of ROI   
            if current_time - last_roi_time >= roi_interval:
                # Save each ROI
                for i, (edges, contour_image, pixel_area, _, area_cm2) in enumerate(roi_results):
                    edge_path = f"roi_images_test{test_id}/{ROI_NAMES[i]}_edges_{timestamp}.jpg"
                    contour_path = f"roi_images_test{test_id}/{ROI_NAMES[i]}_contours_{timestamp}.jpg"
                    
                    #resize the taken images from camera
                    resized_edge = resized_frame(edges)
                    resized_pic = resized_frame(contour_image)
                    
                    cv2.imwrite(edge_path, resized_edge)
                    cv2.imwrite(contour_path, resized_pic)
                    last_screenshot_time = current_time
                    print(f"ROI_Screenshots captured at {timestamp}")

            # Data capture for CSV every 10 seconds
            if current_time - last_capture_time >= interval:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                elapsed_time = current_time - start_time
                
                # Save ROI data
                row_data = [timestamp, f"{elapsed_time:.2f}"]
                
                # Add each ROI's data to CSV row
                for i, (_, _, pixel_area, _, area_cm2) in enumerate(roi_results):
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