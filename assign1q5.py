from PIL import Image
import numpy as np


# Function to check if image matches Poland flag
def is_poland_flag(image):
    # Convert image to RGB and then to numpy array
    img_rgb = image.convert("RGB")
    img_array = np.array(img_rgb)

    # Split image into two horizontal parts (top half and bottom half)
    height, width, _ = img_array.shape
    top_half = img_array[:height // 2, :]
    bottom_half = img_array[height // 2:, :]

    # Check color of top and bottom halves (for Poland: white top, red bottom)
    top_color = np.mean(top_half, axis=(0, 1))  # Get average color of the top half
    bottom_color = np.mean(bottom_half, axis=(0, 1))  # Get average color of the bottom half

    # Define thresholds for white and red colors (using RGB values)
    white_threshold = np.array([230, 230, 230])  # RGB values close to white
    red_threshold = np.array([200, 0, 0])  # RGB values close to red

    # Check if the top half is white and the bottom half is red
    if np.all(top_color > white_threshold) and np.all(bottom_color > red_threshold):
        return True
    return False


# Function to check if image matches Indonesia flag
def is_indonesia_flag(image):
    # Convert image to RGB and then to numpy array
    img_rgb = image.convert("RGB")
    img_array = np.array(img_rgb)

    # Split image into two horizontal parts (top half and bottom half)
    height, width, _ = img_array.shape
    top_half = img_array[:height // 2, :]
    bottom_half = img_array[height // 2:, :]

    # Check color of top and bottom halves (for Indonesia: red top, white bottom)
    top_color = np.mean(top_half, axis=(0, 1))  # Get average color of the top half
    bottom_color = np.mean(bottom_half, axis=(0, 1))  # Get average color of the bottom half

    # Define thresholds for red and white colors (using RGB values)
    red_threshold = np.array([200, 0, 0])  # RGB values close to red
    white_threshold = np.array([230, 230, 230])  # RGB values close to white

    # Check if the top half is red and the bottom half is white
    if np.all(top_color > red_threshold) and np.all(bottom_color > white_threshold):
        return True
    return False


# Main function to check the flag
def check_flag(image_path):
    # Open image using Pillow
    image = Image.open(image_path)

    # Resize image to a standard size (optional, can be removed)
    image = image.resize((400, 200))  # Resize for better comparison

    # Check if the image is the flag of Poland or Indonesia
    if is_poland_flag(image):
        print("The image is the flag of Poland.")
    elif is_indonesia_flag(image):
        print("The image is the flag of Indonesia.")
    else:
        print("The image is neither the flag of Poland nor Indonesia.")


# Test with user input image
image_path = input("Enter the path to the image: ")
check_flag(image_path)
