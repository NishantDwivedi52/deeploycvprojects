from PIL import Image
import numpy as np

def find_color_centers(image_path): #use image path without semi colon
    try:
        # Open the image and convert it to RGB format
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Get the dimensions of the image
        height, width, _ = img_array.shape

        # Flatten the image to analyze all pixels
        pixel_coords = np.indices((height, width)).reshape(2, -1).T
        pixels = img_array.reshape(-1, 3)

        # Define thresholds for red and white
        red_threshold = [150, 0, 0]
        white_threshold = [200, 200, 200]

        # Identify red and white pixels
        red_pixels = np.all(pixels > red_threshold, axis=1)
        white_pixels = np.all(pixels > white_threshold, axis=1)

        # Find the y-coordinates (row values) of red and white pixels
        red_y_coords = pixel_coords[red_pixels][:, 0]
        white_y_coords = pixel_coords[white_pixels][:, 0]

        # Calculate the average y-coordinates (vertical center) for red and white
        red_center_y = np.mean(red_y_coords) if red_y_coords.size > 0 else None
        white_center_y = np.mean(white_y_coords) if white_y_coords.size > 0 else None

        return red_center_y, white_center_y

    except Exception as e:
        return f"Error: {e}"

def determine_flag(image_path):
    red_center, white_center = find_color_centers(image_path)

    if red_center and white_center:
        # Compare the vertical positions of red and white centers
        if red_center < white_center:
            return "The flag is of Indonesia (Red on top, White on bottom)."
        elif white_center < red_center:
            return "The flag is of Poland (White on top, Red on bottom)."
        else:
            return "Cannot determine flag; centers are at the same height."
    else:
        return "Could not detect sufficient red or white areas to identify the flag."

# Test the function with an image
image_path = input("Enter the path to the image: ")
result = determine_flag(image_path)
print(result)

    


 
