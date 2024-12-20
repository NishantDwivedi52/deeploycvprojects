import torch
import cv2
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import functional as F
import numpy as np

# Step 1: Load the Pretrained SSD Model
weights = SSD300_VGG16_Weights.COCO_V1  # Use COCO pretrained weights
model = ssd300_vgg16(weights=weights)
model.eval()  # Set the model to evaluation mode

# Expanded mapping for country flags (added as many as possible)
country_flags = {
    0: 'background',
    1: 'Flag of Afghanistan',
    2: 'Flag of Albania',
    3: 'Flag of Algeria',
    4: 'Flag of Andorra',
    5: 'Flag of Angola',
    6: 'Flag of Antigua and Barbuda',
    7: 'Flag of Argentina',
    8: 'Flag of Armenia',
    9: 'Flag of Australia',
    10: 'Flag of Austria',
    11: 'Flag of Azerbaijan',
    12: 'Flag of Bahamas',
    13: 'Flag of Bahrain',
    14: 'Flag of Bangladesh',
    15: 'Flag of Barbados',
    16: 'Flag of Belarus',
    17: 'Flag of Belgium',
    18: 'Flag of Belize',
    19: 'Flag of Benin',
    20: 'Flag of Bhutan',
    21: 'Flag of Bolivia',
    22: 'Flag of Bosnia and Herzegovina',
    23: 'Flag of Botswana',
    24: 'Flag of Brazil',
    25: 'Flag of Brunei Darussalam',
    26: 'Flag of Bulgaria',
    27: 'Flag of Burkina Faso',
    28: 'Flag of Burundi',
    29: 'Flag of Cabo Verde',
    30: 'Flag of Cambodia',
    31: 'Flag of Cameroon',
    32: 'Flag of Canada',
    33: 'Flag of Central African Republic',
    34: 'Flag of Chad',
    35: 'Flag of Chile',
    36: 'Flag of China',
    37: 'Flag of Colombia',
    38: 'Flag of Comoros',
    39: 'Flag of Congo',
    40: 'Flag of Costa Rica',
    41: 'Flag of Croatia',
    42: 'Flag of Cuba',
    43: 'Flag of Cyprus',
    44: 'Flag of Czechia',
    45: 'Flag of Democratic Republic of the Congo',
    46: 'Flag of Denmark',
    47: 'Flag of Djibouti',
    48: 'Flag of Dominica',
    49: 'Flag of Dominican Republic',
    50: 'Flag of Ecuador',
    51: 'Flag of Egypt',
    52: 'Flag of El Salvador',
    53: 'Flag of Equatorial Guinea',
    54: 'Flag of Eritrea',
    55: 'Flag of Estonia',
    56: 'Flag of Eswatini',
    57: 'Flag of Ethiopia',
    58: 'Flag of Fiji',
    59: 'Flag of Finland',
    60: 'Flag of France',
    61: 'Flag of Gabon',
    62: 'Flag of Gambia',
    63: 'Flag of Georgia',
    64: 'Flag of Germany',
    65: 'Flag of Ghana',
    66: 'Flag of Greece',
    67: 'Flag of Grenada',
    68: 'Flag of Guatemala',
    69: 'Flag of Guinea',
    70: 'Flag of Guinea-Bissau',
    71: 'Flag of Guyana',
    72: 'Flag of Haiti',
    73: 'Flag of Honduras',
    74: 'Flag of Hungary',
    75: 'Flag of Iceland',
    76: 'Flag of India',
    77: 'Flag of Indonesia',
    78: 'Flag of Iran',
    79: 'Flag of Iraq',
    80: 'Flag of Ireland',
    81: 'Flag of Israel',
    82: 'Flag of Italy',
    83: 'Flag of Ivory Coast',
    84: 'Flag of Jamaica',
    85: 'Flag of Japan',
    86: 'Flag of Jordan',
    87: 'Flag of Kazakhstan',
    88: 'Flag of Kenya',
    89: 'Flag of Kiribati',
    90: 'Flag of Korea (North)',
    91: 'Flag of Korea (South)',
    92: 'Flag of Kuwait',
    93: 'Flag of Kyrgyzstan',
    94: 'Flag of Laos',
    95: 'Flag of Latvia',
    96: 'Flag of Lebanon',
    97: 'Flag of Lesotho',
    98: 'Flag of Liberia',
    99: 'Flag of Libya',
    # Add more flags here...
    # Add all other countries you need
}

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
    for box, score, label in zip(detections[0]['boxes'], detections[0]['scores'], detections[0]['labels']):
        if score > threshold:
            # Convert coordinates from normalized to original size
            x1, y1, x2, y2 = box.tolist()

            # Rescale bounding box coordinates to original image size
            x1, y1, x2, y2 = int(x1 * original_width / 300), int(y1 * original_height / 300), int(
                x2 * original_width / 300), int(y2 * original_height / 300)

            # Get label text from the dictionary
            label_text = f"{country_flags.get(label.item(), 'Unknown')}: {score:.2f}"

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

# Step 5: Process the Image
def process_image(input_image_path, output_image_path, model):
    """Process the image, apply SSD, and save the output."""
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Could not open image.")
        return

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Detect objects
    detections = detect_objects(model, image_tensor)

    # Draw detections on the image
    output_image = draw_detections(image, detections, original_width, original_height)

    # Save the output image
    cv2.imwrite(output_image_path, output_image)
    print(f"Processing completed. Output saved to: {output_image_path}")

    # Optionally, display the image
  #  cv2.imshow("Flag Detection", output_image)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()

# Step 6: Run the SSD Detection on Your Image
if __name__ == "__main__":
    input_image_path = "indonesianflag2.jpg"  # Path to your input image (flag image)
    output_image_path = "indofinal.jpg"  # Path for the output image

    process_image(input_image_path, output_image_path, model)
