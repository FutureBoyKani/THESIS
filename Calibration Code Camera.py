import numpy as np
import cv2 as cv
import glob
import os
import time
import argparse
from picamera2 import Picamera2

def capture_calibration_images(num_images=20, delay=2, chessboard_size=(8, 6), output_dir="calibration_images"):
    """
    Capture images of a chessboard for camera calibration
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the camera
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    
    print(f"Starting image capture for calibration. Hold a {chessboard_size[0]}x{chessboard_size[1]} chessboard in front of the camera.")
    print(f"Will capture {num_images} images with {delay} seconds delay between each capture.")
    
    # Allow camera to initialize
    time.sleep(2)
    
    for i in range(num_images):
        print(f"Capturing image {i+1}/{num_images} in 3 seconds...")
        time.sleep(3)
        
        # Capture image
        image = picam2.capture_array()
        
        # Convert from RGB to BGR (OpenCV format)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        # Save the image
        filename = f"{output_dir}/chessboard_{i+1:02d}.jpg"
        cv.imwrite(filename, image)
        print(f"Saved {filename}")
        
        time.sleep(delay)
    
    picam2.stop()
    print("Finished capturing calibration images.")

def calibrate_camera(chessboard_size=(8, 6), square_size=1.0, image_dir="calibration_images", output_file="camera_calibration.npz"):
    """
    Calibrate camera using chessboard images
    """
    print("Starting camera calibration process...")
    
    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    images = glob.glob(f'{image_dir}/*.jpg')
    
    if not images:
        print(f"No images found in {image_dir}. Please capture calibration images first.")
        return False
    
    print(f"Found {len(images)} images for calibration.")
    
    # Process each image
    successful_images = 0
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, add object points and image points
        if ret:
            successful_images += 1
            objpoints.append(objp)
            
            # Refine corner locations
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Display the corners
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow('Chessboard Corners', img)
            cv.waitKey(500)
    
    cv.destroyAllWindows()
    
    if successful_images < 10:
        print(f"Only found chessboard corners in {successful_images} images. Need at least 10 for good calibration.")
        return False
    
    print(f"Successfully processed {successful_images} images.")
    
    # Calibrate camera
    print("Calculating calibration parameters...")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Save the calibration results
    np.savez(output_file, camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"Calibration complete! Parameters saved to {output_file}")
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    print(f"Total reprojection error: {mean_error/len(objpoints)}")
    return True

def undistort_test(image_path, calibration_file="camera_calibration.npz"):
    """
    Test the calibration parameters by undistorting an image
    """
    # Load calibration parameters
    if not os.path.exists(calibration_file):
        print(f"Calibration file {calibration_file} not found.")
        return
    
    with np.load(calibration_file) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    
    # Load image
    img = cv.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort
    dst = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # Display results
    cv.imshow('Original Image', img)
    cv.imshow('Undistorted Image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi 5 Camera Calibration')
    
    parser.add_argument('--mode', type=str, choices=['capture', 'calibrate', 'test'], required=True,
                      help='Operation mode: capture images, calibrate camera, or test calibration')
    
    parser.add_argument('--num_images', type=int, default=20,
                      help='Number of calibration images to capture')
    
    parser.add_argument('--delay', type=int, default=2,
                      help='Delay between image captures in seconds')
    
    parser.add_argument('--chessboard_rows', type=int, default=6,
                      help='Number of internal corners in the chessboard rows')
    
    parser.add_argument('--chessboard_cols', type=int, default=9,
                      help='Number of internal corners in the chessboard columns')
    
    parser.add_argument('--square_size', type=float, default=1.0,
                      help='Size of chessboard squares in your preferred unit')
    
    parser.add_argument('--image_dir', type=str, default='calibration_images',
                      help='Directory for calibration images')
    
    parser.add_argument('--calibration_file', type=str, default='camera_calibration.npz',
                      help='File to save/load calibration parameters')
    
    parser.add_argument('--test_image', type=str,
                      help='Image to test calibration parameters on')
    
    args = parser.parse_args()
    
    if args.mode == 'capture':
        capture_calibration_images(
            num_images=args.num_images,
            delay=args.delay,
            chessboard_size=(args.chessboard_cols, args.chessboard_rows),
            output_dir=args.image_dir
        )
    
    elif args.mode == 'calibrate':
        calibrate_camera(
            chessboard_size=(args.chessboard_cols, args.chessboard_rows),
            square_size=args.square_size,
            image_dir=args.image_dir,
            output_file=args.calibration_file
        )
    
    elif args.mode == 'test':
        if not args.test_image:
            print("Please provide a test image with --test_image")
            return
        undistort_test(args.test_image, args.calibration_file)

if __name__ == "__main__":
    main()