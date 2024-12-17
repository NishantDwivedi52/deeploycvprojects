import cv2
import numpy as np
import matplotlib.pyplot as plt

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera")
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("Failed to capture image")
        return None

def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def thresholding_image(img, threshold_value=127):
    _, thresholded = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

def quantize_16_gray(img):
    quantized = (img // 16) * 16
    return quantized

def sobel_filter(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    return sobel_edges

def canny_edge_detection(img):
    return cv2.Canny(img, 100, 200)

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def show_images_in_grid(images):
    plt.figure(figsize=(10, 5))
    for i, (title, img) in enumerate(images.items()):
        plt.subplot(2, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main execution
img = capture_image()

if img is not None:
    gray_img = grayscale_image(img)
    thresh_img = thresholding_image(gray_img)
    quantized_img = quantize_16_gray(gray_img)
    sobel_img = sobel_filter(gray_img)
    canny_img = canny_edge_detection(gray_img)
    blurred_img = gaussian_blur(gray_img)
    sharpened_img = sharpen_image(blurred_img)
    rgb_bgr_img = rgb_to_bgr(img)

    # Store images with titles
    images = {
        "Gray": gray_img,
        "Thresholded": thresh_img,
        "Quantized": quantized_img,
        "Sobel": sobel_img,
        "Canny": canny_img,
        "Gaussian Blur": blurred_img,
        "Sharpened": sharpened_img,
        "RGB to BGR": rgb_bgr_img
    }

    show_images_in_grid(images)
