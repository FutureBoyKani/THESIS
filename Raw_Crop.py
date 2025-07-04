import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import datetime
import rawpy
import pandas as pd
import glob

# Initialize camera with RAW configuration
settings = {
    "ColourGains": (2.68, 1.478),
    "AwbEnable": False,
    "AeEnable": False,
    "AfMode": 0,  # Manual mode
    "LensPosition": 10.0,
    "Sharpness": 1.0,
    "Saturation": 1.0,
    "Contrast": 1.0,
    "Brightness": 0.0,
    "AnalogueGain": 1.12,
    "ExposureTime": 2600
}

def crop_image(img_display_copy): # Renamed 'img' to 'img_display_copy' for clarity
    """
    Detects four circles in an image and calculates a rectangular ROI
    based on their sorted positions. It also draws the detected circles
    and the calculated ROI on the input image for visualization.

    Args:
        img_display_copy (numpy.ndarray): A copy of the image on which circles and ROI
                                          will be drawn for display.

    Returns:
        list: [x, y, width, height] of the calculated ROI, or None if not enough
              circles are detected or no circles are found.
    """
    gray = cv2.cvtColor(img_display_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Parameters for HoughCircles can be tuned.
    # dp: inverse ratio of the accumulator resolution to the image resolution.
    # minDist: minimum distance between the centers of the detected circles.
    # param1: higher threshold for the Canny edge detector (used internally).
    # param2: accumulator threshold for the circle centers.
    # minRadius/maxRadius: range of circle radii to detect.
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=50, param2=30, minRadius=0, maxRadius=40)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Ensure we have at least 4 circles to define the rectangle
        if len(circles[0]) >= 4:
            # Convert circle coordinates to points (x, y)
            # We take the first 4 circles, assuming they correspond to the corners
            points = [(x, y) for (x, y, r) in circles[0][:4]]
            
            # Sort points by x coordinate first
            points.sort(key=lambda p: p[0])
            
            # Split into left and right pairs
            left_pair = sorted(points[:2], key=lambda p: p[1])  # Sort left by y
            right_pair = sorted(points[2:], key=lambda p: p[1]) # Sort right by y
            
            # Extract corner points (assuming roughly rectangular arrangement)
            top_left = left_pair[0]
            bottom_left = left_pair[1]
            top_right = right_pair[0]
            bottom_right = right_pair[1]
            
            # Calculate ROI coordinates based on these four points
            # We take the minimum x and y for top-left, and maximum x and y for bottom-right
            # This ensures the rectangle encloses all four points
            x = bottom_left[0] 
            y = top_left[1] - 30  # Use the y of the top-left point
            width = bottom_right[0] - bottom_left[0] 
            height = 50 # Fixed height for the rectangle
            
            # Add some padding to the ROI to ensure it fully encompasses the detected circles
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = width + 2 * padding
            height = height + 2 * padding
            
            # --- Visualization on the `img_display_copy` ---
            # Draw detected circles
            for circle in circles[0][:4]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(img_display_copy, center, radius, (0, 255, 0), 2)
            
            # Draw ROI rectangle on the `img_display_copy`
            cv2.rectangle(img_display_copy, (x, y), (x + width, y + height), (0, 0, 255), 2) # Red rectangle
            # --- End Visualization ---

            return [x, y, width, height]
            
        else:
            print(f"Not enough circles detected. Found {len(circles[0])} circles, need 4.")
            return None
    else:
        print("No circles were detected.")
        return None


def adjust_to_r_pixel(coords):
    """Adjust coordinates to start from R pixel in RGGB pattern"""
    x, y, w, h = coords
    # Ensure x starts on an even column (R pixel)
    if x % 2 != 0:
        x -= 1
    # Ensure y starts on an even row (R pixel)
    if y % 2 != 0:
        y -= 1
    return [y, x, h, w]  # Return as [y, x, h, w] for array indexing

def process_dng_file(file_path, crop_coords):
    """
    Process DNG file and extract RGB channels with proper handling of Bayer pattern
    Returns cropped RGB channels with zeros replaced by NaN
    """
    try:
        with rawpy.imread(file_path) as raw:
            # Get raw data and apply black level correction
            raw_data = raw.raw_image_visible.copy()
            raw_data = np.maximum(raw_data - 4096, 0)
            pattern = raw.raw_pattern
            
            # Get dimensions
            height, width = raw_data.shape

            # Create masks for R/G/B channels
            r_mask = np.zeros((height, width), dtype=bool)
            g_mask = np.zeros_like(r_mask)
            b_mask = np.zeros_like(r_mask)

            # Create index grids for Bayer pattern
            rows, cols = np.indices((height, width))
            mod_rows = rows % 2
            mod_cols = cols % 2

            # Generate masks based on Bayer pattern
            for r in range(2):
                for c in range(2):
                    color_code = pattern[r, c]
                    mask = (mod_rows == r) & (mod_cols == c)
                    if color_code == 0:  # Red
                        r_mask[mask] = True
                    elif color_code == 1:  # Green
                        g_mask[mask] = True
                    elif color_code == 2:  # Blue
                        b_mask[mask] = True
                    elif color_code == 3:  # Second green pixel
                        g_mask[mask] = True

            # Create and fill channel arrays
            r_channel = np.zeros_like(raw_data)
            g_channel = np.zeros_like(raw_data)
            b_channel = np.zeros_like(raw_data)

            r_channel[r_mask] = raw_data[r_mask]
            g_channel[g_mask] = raw_data[g_mask]
            b_channel[b_mask] = raw_data[b_mask]

            # Crop channels
            y, x, h, w = crop_coords
            r_cropped = r_channel[y:y+h, x:x+w]
            g_cropped = g_channel[y:y+h, x:x+w]
            b_cropped = b_channel[y:y+h, x:x+w]

            # Replace zeros with NaN
            r_cropped = np.where(r_cropped == 0, np.nan, r_cropped)
            g_cropped = np.where(g_cropped == 0, np.nan, g_cropped)
            b_cropped = np.where(b_cropped == 0, np.nan, b_cropped)

            return r_cropped, g_cropped, b_cropped
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def extract_timestamp_from_filename(filename):
    """Extract datetime from filename like '20250528_143022_full'"""
    try:
        # Extract timestamp part (first two underscore-separated parts)
        parts = filename.split('_')
        if len(parts) >= 2:
            timestamp_str = f"{parts[0]}_{parts[1]}"
            return pd.to_datetime(timestamp_str, format='%Y%m%d_%H%M%S')
    except:
        pass
    return None

def process_all_dng_files(test_dir, crop_coords):
    """
    Process all DNG files in the raw_captures directory and create time series CSV files 
    with each column representing a path length for each color
    """
    raw_captures_dir = os.path.join(test_dir, "images", "raw_captures")
    csv_dir = os.path.join(test_dir, "data", "csv")
    
    # Get all DNG files
    dng_files = glob.glob(os.path.join(raw_captures_dir, "*.dng"))
    dng_files.sort()
    
    if not dng_files:
        print(f"No DNG files found in {raw_captures_dir}")
        return
    
    print(f"Found {len(dng_files)} DNG files to process")
    
    # Lists to store data for each color
    red_data = []
    green_data = []
    blue_data = []
    elapsed_times = []
    
    # Variable to store the first timestamp for elapsed time calculation
    first_timestamp = None
    
    # Process all files
    for i, file_path in enumerate(dng_files):
        filename = os.path.basename(file_path).replace('.dng', '')
        print(f"Processing {filename} ({i+1}/{len(dng_files)})")
        
        # Extract timestamp
        timestamp = extract_timestamp_from_filename(filename)
        if timestamp is None:
            print(f"Warning: Could not parse timestamp from {filename}")
            continue
        
        # Set first timestamp for elapsed time calculation
        if first_timestamp is None:
            first_timestamp = timestamp
        
        # Calculate elapsed time in seconds
        elapsed_seconds = (timestamp - first_timestamp).total_seconds()
        
        r_cropped, g_cropped, b_cropped = process_dng_file(file_path, crop_coords)
        
        if r_cropped is None:
            continue
        
        # Process each channel - get mean intensities across height for each column (path length)
        r_mean = np.nanmean(r_cropped, axis=0)
        g_mean = np.nanmean(g_cropped, axis=0)
        b_mean = np.nanmean(b_cropped, axis=0)
        
        # Store data
        elapsed_times.append(elapsed_seconds)
        red_data.append(r_mean)
        green_data.append(g_mean)
        blue_data.append(b_mean)
    
    # Create path length values
    pathlength_values = np.linspace(1, 10, r_cropped.shape[1])  
    
    # Create column names for path lengths
    pathlength_columns = [f"pathlength_{pl:.2f}" for pl in pathlength_values]
    
    # Create DataFrames for each color
    if red_data:
        # Red DataFrame
        red_df = pd.DataFrame(red_data, columns=pathlength_columns)
        red_df.insert(0, 'elapsed_seconds', elapsed_times)
        
        # Green DataFrame
        green_df = pd.DataFrame(green_data, columns=pathlength_columns)
        green_df.insert(0, 'elapsed_seconds', elapsed_times)
        
        # Blue DataFrame
        blue_df = pd.DataFrame(blue_data, columns=pathlength_columns)
        blue_df.insert(0, 'elapsed_seconds', elapsed_times)
        
        # Save CSV files
        try:
            red_path = os.path.join(csv_dir, "red_intensity_timeseries.csv")
            green_path = os.path.join(csv_dir, "green_intensity_timeseries.csv")
            blue_path = os.path.join(csv_dir, "blue_intensity_timeseries.csv")
            
            red_df.to_csv(red_path, index=False)
            green_df.to_csv(green_path, index=False)
            blue_df.to_csv(blue_path, index=False)
            
            print(f"Successfully saved CSV files:")
            print(f"  Red: {red_path}")
            print(f"  Green: {green_path}")
            print(f"  Blue: {blue_path}")
        except Exception as e:
            print(f"Error saving CSV files: {e}")
    
    print("DNG processing complete!")

def setup_directories(test_id):
    """Set up all needed directories"""
    test_name = input("Enter test name: ")
    test_dir = f"{test_name}_test_{test_id}"
    
    subdirs = [
        "images/raw_captures",
        "data/csv"
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(test_dir, subdir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            
    return test_dir, test_name

def main(test_id="001"):
    picam2 = None
    try:
        # Setup directories
        test_dir, test_name = setup_directories(test_id)
        
        picam2 = Picamera2()
        capture_config = picam2.create_still_configuration(
            raw={"size": picam2.sensor_resolution},  # RAW capture
            main={"size": (4608, 2592)},            # Display frame
            controls={"AfMode": 0}                  
        )
        preview_config = picam2.create_preview_configuration()
        picam2.configure(preview_config)
        
        print("Starting RAW capture. Controls:")
        print("'q' - quit and process DNG files")
        print("'s' - save current frame")
        print("'p' - process existing DNG files")
        print("'r' - reset ROI detection and re-detect on next frame") # Added 'r' option
        
        # Initialize timing
        interval = 120
        last_capture = 0  
        # Initialize ROI coordinates to None; it will be detected on the first frame
        roi_coords = None  
        adjusted_coords = None # To store the R-pixel adjusted coordinates for DNG processing
        
        picam2.start()
        
        # Apply eye-best settings
        picam2.set_controls(settings)
        print("Applied eye-best settings")
        
        time.sleep(2)  # Allow camera to warm up
        
        while True:
            current_time = time.time()
            
            # Capture RAW frame
            request = picam2.switch_mode_and_capture_request(capture_config)
            display_frame = request.make_array("main")
            
            # --- Main change: Use crop_image to find and draw ROI ---
            if roi_coords is None:
                # Pass a copy of display_frame to crop_image so it can draw on it
                # without affecting the original frame used for processing later
                roi_coords = crop_image(display_frame.copy()) 
                if roi_coords is not None:
                    # If ROI is successfully detected, calculate adjusted_coords once
                    adjusted_coords = adjust_to_r_pixel(roi_coords)
                    print(f"ROI detected: {roi_coords}. Adjusted for R-pixel: {adjusted_coords}")
                else:
                    print("ROI could not be detected. Please ensure 4 distinct circles are visible.")
            
            # If ROI is detected, draw it on the current display_frame
            if roi_coords is not None:
                x, y, w, h = roi_coords
                
                # Draw rectangle for the exact analysis area
                cv2.rectangle(display_frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0), 3) # Green rectangle
                
                # Add red dot at the top-left corner
                cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1) # Red dot
                
                # Add label
                label_y = max(15, y - 10) # Ensure label is visible at the top
                cv2.putText(display_frame, f"Analysis ROI ({x},{y})",
                          (x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # --- End Main Change ---
            
            # Auto-capture at intervals (only if ROI has been detected)
            if current_time - last_capture >= interval and adjusted_coords is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_path = os.path.join(test_dir, "images/raw_captures", f"{timestamp}_full.dng")
                request.save_dng(raw_path)
                print(f"Auto-saved: {timestamp}_full.dng")
                last_capture = current_time
            
            # Display
            display_frame_resized = cv2.resize(display_frame, (1920, 1080)) # Resize for display
            cv2.imshow("Main View", display_frame_resized)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting and processing DNG files...")
                break
            elif key == ord('s') and adjusted_coords is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_path = os.path.join(test_dir, "images/raw_captures", f"{timestamp}_full.dng")
                request.save_dng(raw_path)
                print(f"Manually saved: {timestamp}_full.dng")
            elif key == ord('p') and adjusted_coords is not None:
                print("Processing existing DNG files...")
                process_all_dng_files(test_dir, adjusted_coords)
            elif key == ord('r'):
                print("Resetting ROI detection...")
                roi_coords = None # Reset roi_coords to trigger re-detection
                adjusted_coords = None
            
            request.release()
        
        # Process DNG files after quitting
        if picam2:
            picam2.stop()
        cv2.destroyAllWindows()
        
        if adjusted_coords is not None: # Ensure adjusted_coords is available for final processing
            print("Processing all captured DNG files...")
            process_all_dng_files(test_dir, adjusted_coords)
        
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
