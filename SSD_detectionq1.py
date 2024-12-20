import cv2
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import functional as F
import numpy as np

# Step 1: Load the Pretrained SSD Model
weights = SSD300_VGG16_Weights.COCO_V1  # Use COCO pretrained weights
model = ssd300_vgg16(weights=weights)
model.eval()  # Set the model to evaluation mode


# Step 2: Preprocess the Input Image
def preprocess_image(image):
    """Convert image to tensor and normalize it."""
    # Convert BGR (OpenCV format) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to 300x300 (SSD input size)
    image_resized = cv2.resize(image_rgb, (300, 300))
    # Convert to tensor and normalize
    image_tensor = F.to_tensor(image_resized)
    image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image_tensor


# Step 3: Run Object Detection
def detect_objects(model, image_tensor):
    """Detect objects in the image using the SSD model."""
    # Add batch dimension (1, C, H, W)
    image_batch = image_tensor.unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        detections = model(image_batch)
    return detections


# Step 4: Draw Bounding Boxes and Labels
def draw_detections(image, detections, original_width, original_height, threshold=0.5):
    """Draw bounding boxes and labels for the detected objects."""
    # COCO class labels
    labels = weights.meta["categories"]

    # Loop through detections and draw boxes with confidence > threshold
    for box, score, label in zip(detections[0]['boxes'], detections[0]['scores'], detections[0]['labels']):
        if score > threshold:
            # Convert coordinates from normalized to original size
            x1, y1, x2, y2 = box.tolist()

            # Rescale bounding box coordinates to original image size
            x1, y1, x2, y2 = int(x1 * original_width / 300), int(y1 * original_height / 300), int(
                x2 * original_width / 300), int(y2 * original_height / 300)

            label_text = f"{labels[label]}: {score:.2f}"

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


# Step 5: Process the Video
def process_video(input_video_path, output_video_path, model):
    """Process the video, apply SSD, and save the output."""
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter to save output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get original frame dimensions
        original_height, original_width = frame.shape[:2]

        # Preprocess the frame
        image_tensor = preprocess_image(frame)

        # Detect objects
        detections = detect_objects(model, image_tensor)

        # Draw detections on the frame
        output_frame = draw_detections(frame, detections, original_width, original_height)

        # Write the processed frame to the output video
        out.write(output_frame)

    # Release resources
    cap.release()
    out.release()
    print("Processing completed. Output saved to:", output_video_path)


# Step 6: Run the SSD Detection on Your Video
if __name__ == "__main__":
    input_video_path = "Testvideo.mp4.mp4"  # Path to your input video
    output_video_path = "output_video.mp4"  # Path for the output video

    process_video(input_video_path, output_video_path, model)
