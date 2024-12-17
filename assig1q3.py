
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images of your choice
img1 = cv2.imread('MrUCL.webp')  # Replace with your image path
img2 = cv2.imread('ThiefMessi.jpeg')  # Replace with your image path
#img1.np.show()
# Convert to grayscale if needed (for simplicity)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Low-pass filter using Gaussian blur with smaller kernel size
def low_pass_filter(img, kernel_size=(15, 15)):
    return cv2.GaussianBlur(img, kernel_size, 0)

# High-pass filter (subtracting low-pass filtered image from the original)
def high_pass_filter(img):
    low_pass = low_pass_filter(img)
    high_pass = img - low_pass
    return high_pass

# Apply high-pass filter on the first image
high_pass_img1 = high_pass_filter(img1_gray)

# Apply low-pass filter on the second image
low_pass_img2 = low_pass_filter(img2_gray)

# Resize high_pass_img1 to match low_pass_img2 size (or vice versa)
high_pass_img1_resized = cv2.resize(high_pass_img1, (low_pass_img2.shape[1], low_pass_img2.shape[0]))

# Combine the high-pass filtered image and low-pass filtered image
combined_img = cv2.add(high_pass_img1_resized, low_pass_img2)

# Normalize combined image to the 0-255 range for better clarity
combined_img = cv2.normalize(combined_img, None, 0, 255, cv2.NORM_MINMAX)

# Display all the images
def show_images_in_grid(images):
    plt.figure(figsize=(15, 10))
    for i, (title, img) in enumerate(images.items()):
        plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Images to display
images = {
    "Original Image 1": img1_gray,
    "High-pass Filter 1": high_pass_img1,
    "Original Image 2": img2_gray,
    "Low-pass Filter 2": low_pass_img2,
    "Combined Image": combined_img
}

# Show images
show_images_in_grid(images)
