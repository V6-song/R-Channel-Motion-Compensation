import cv2
import numpy as np
import os
import logging
import shutil
from option import opt

# Set up logging to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler to log to a file
folder_path = "./test"
folder_name = os.path.basename(folder_path)

# Ensure the logs directory exists
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

# Create the file handler with the folder name as the log file name
file_handler = logging.FileHandler(os.path.join(log_dir, f'{folder_name}.txt'))
file_handler.setLevel(logging.INFO)

# Create a stream handler to log to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Define the log format
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def filter_outliers(data, threshold=1.5):
    """Filter outliers based on the interquartile range (IQR)."""
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def calculate_average_displacement(img1, img2, num_keypoints=1000, filter_outlier=True):
    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    if len(matches) == 0:
        raise ValueError("No matches found between the two images.")

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use top N matches
    selected_matches = matches[:min(num_keypoints, len(matches))]

    displacements = []

    for match in selected_matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        displacement = np.linalg.norm(np.array(pt1) - np.array(pt2))
        displacements.append(displacement)

    # Filter outliers if enabled
    if filter_outlier:
        displacements = filter_outliers(displacements)

    # Calculate average displacement
    average_displacement = np.mean(displacements) if len(displacements) > 0 else 0

    return average_displacement, selected_matches, keypoints1, keypoints2

def find_optimal_num_keypoints(img1, img2, max_keypoints=1000, step=50):
    """Find the num_keypoints value that minimizes the average displacement."""
    optimal_keypoints = 0
    minimal_displacement = float('inf')

    for num_keypoints in range(step, max_keypoints + 1, step):
        try:
            avg_disp, _, _, _ = calculate_average_displacement(img1, img2, num_keypoints=num_keypoints)
            if avg_disp < minimal_displacement:
                minimal_displacement = avg_disp
                optimal_keypoints = num_keypoints
        except ValueError:
            continue  # Skip if no matches found

    return optimal_keypoints, minimal_displacement
    
def process_image_pairs(folder_path):
    hazy_folder = os.path.join(folder_path, 'S_r_q')
    clear_folder = os.path.join(folder_path, 'C_r_star')

    if not os.path.isdir(hazy_folder) or not os.path.isdir(clear_folder):
        raise FileNotFoundError("Either 'S_r_q' or 'C_r_star' folder does not exist in the specified path.")

    hazy_images = sorted(os.listdir(hazy_folder), key=lambda x: int(x[-11:-4]))
    clear_images = sorted(os.listdir(clear_folder), key=lambda x: int(x[-11:-4]))

    # Find unmatched images
    hazy_only = set(hazy_images) - set(clear_images)
    clear_only = set(clear_images) - set(hazy_images)

    if hazy_only:
        logger.info("The following hazy images do not have corresponding clear images:")
        for image_name in hazy_only:
            logger.info(f"  {image_name}")
    if clear_only:
        logger.info("The following clear images do not have corresponding hazy images:")
        for image_name in clear_only:
            logger.info(f"  {image_name}")

    # Ensure paired images by matching filenames
    paired_images = set(hazy_images).intersection(clear_images)

    if len(paired_images) == 0:
        raise ValueError("No paired images found between 'S_r_q' and 'C_r_star' folders.")

    paired_images = sorted(paired_images, key=lambda x: int(x[-11:-4]))

    # Create under1 and above10 folders and subfolders
    under1_folder = os.path.join(folder_path, 'under1')
    hazy_under1_folder = os.path.join(under1_folder, 'S_r_q')
    clear_under1_folder = os.path.join(under1_folder, 'C_r_star')
    os.makedirs(hazy_under1_folder, exist_ok=True)
    os.makedirs(clear_under1_folder, exist_ok=True)

    above10_folder = os.path.join(folder_path, 'above10')
    hazy_above10_folder = os.path.join(above10_folder, 'S_r_q')
    clear_above10_folder = os.path.join(above10_folder, 'C_r_star')
    os.makedirs(hazy_above10_folder, exist_ok=True)
    os.makedirs(clear_above10_folder, exist_ok=True)

    for image_name in paired_images:
        hazy_path = os.path.join(hazy_folder, image_name)
        clear_path = os.path.join(clear_folder, image_name)

        img1 = cv2.imread(hazy_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(clear_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            logger.warning(f"Skipping {image_name} due to loading error.")
            continue

        try:
            optimal_keypoints, minimal_displacement = find_optimal_num_keypoints(img1, img2)
            avg_disp, matches, kp1, kp2 = calculate_average_displacement(img1, img2, num_keypoints=optimal_keypoints)
            logger.info("----------------------------------------")
            logger.info(f"{image_name}")
            logger.info(f"{avg_disp:.2f} pixels")
            # logger.info(f"Optimal Number of Keypoints: {optimal_keypoints}")
            logger.info(f"Matched points: {len(matches)}")

            # Move images to under1 folder if displacement < 1 pixel
            if avg_disp < 1:
                shutil.move(hazy_path, os.path.join(hazy_under1_folder, image_name))
                shutil.move(clear_path, os.path.join(clear_under1_folder, image_name))
                logger.info(f"Moved {image_name} to under1 folder due to low displacement.")
            # Move images to above10 folder if displacement > 10 pixels
            elif avg_disp > 5:
                shutil.move(hazy_path, os.path.join(hazy_above10_folder, image_name))
                shutil.move(clear_path, os.path.join(clear_above10_folder, image_name))
                logger.info(f"Moved {image_name} to above5 folder due to high displacement.")

        except ValueError as e:
            logger.error(f"Error processing {image_name}: {e}")

# Example usage

folder_path = opt.results_dir  # Path to the folder containing 'S_r_q' and 'C_r_star'
process_image_pairs(folder_path)