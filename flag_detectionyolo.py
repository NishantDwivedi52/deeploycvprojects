import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (use your custom-trained model on flags)
model = YOLO('yolov5s.pt')  # Replace with the path to your trained model

# Define a dictionary for all countries (these IDs should correspond to the trained model)
flags = {
    0: "Afghanistan",
    1: "Albania",
    2: "Algeria",
    3: "Andorra",
    4: "Angola",
    5: "Antigua and Barbuda",
    6: "Argentina",
    7: "Armenia",
    8: "Australia",
    9: "Austria",
    10: "Azerbaijan",
    11: "Bahamas",
    12: "Bahrain",
    13: "Bangladesh",
    14: "Barbados",
    15: "Belarus",
    16: "Belgium",
    17: "Belize",
    18: "Benin",
    19: "Bhutan",
    20: "Bolivia",
    21: "Bosnia and Herzegovina",
    22: "Botswana",
    23: "Brazil",
    24: "Brunei",
    25: "Bulgaria",
    26: "Burkina Faso",
    27: "Burundi",
    28: "Cabo Verde",
    29: "Cambodia",
    30: "Cameroon",
    31: "Canada",
    32: "Central African Republic",
    33: "Chad",
    34: "Chile",
    35: "China",
    36: "Colombia",
    37: "Comoros",
    38: "Congo (Congo-Brazzaville)",
    39: "Congo (Congo-Kinshasa)",
    40: "Costa Rica",
    41: "Croatia",
    42: "Cuba",
    43: "Cyprus",
    44: "Czechia (Czech Republic)",
    45: "Denmark",
    46: "Djibouti",
    47: "Dominica",
    48: "Dominican Republic",
    49: "Ecuador",
    50: "Egypt",
    51: "El Salvador",
    52: "Equatorial Guinea",
    53: "Eritrea",
    54: "Estonia",
    55: "Eswatini (fmr. 'Swaziland')",
    56: "Ethiopia",
    57: "Fiji",
    58: "Finland",
    59: "France",
    60: "Gabon",
    61: "Gambia",
    62: "Georgia",
    63: "Germany",
    64: "Ghana",
    65: "Greece",
    66: "Grenada",
    67: "Guatemala",
    68: "Guinea",
    69: "Guinea-Bissau",
    70: "Guyana",
    71: "Haiti",
    72: "Honduras",
    73: "Hungary",
    74: "Iceland",
    75: "India",
    76: "Indonesia",
    77: "Iran",
    78: "Iraq",
    79: "Ireland",
    80: "Israel",
    81: "Italy",
    82: "Ivory Coast",
    83: "Jamaica",
    84: "Japan",
    85: "Jordan",
    86: "Kazakhstan",
    87: "Kenya",
    88: "Kiribati",
    89: "Korea, North",
    90: "Korea, South",
    91: "Kuwait",
    92: "Kyrgyzstan",
    93: "Laos",
    94: "Latvia",
    95: "Lebanon",
    96: "Lesotho",
    97: "Liberia",
    98: "Libya",
    99: "Liechtenstein",
    100: "Lithuania",
    101: "Luxembourg",
    102: "Madagascar",
    103: "Malawi",
    104: "Malaysia",
    105: "Maldives",
    106: "Mali",
    107: "Malta",
    108: "Marshall Islands",
    109: "Mauritania",
    110: "Mauritius",
    111: "Mexico",
    112: "Micronesia",
    113: "Moldova",
    114: "Monaco",
    115: "Mongolia",
    116: "Montenegro",
    117: "Morocco",
    118: "Mozambique",
    119: "Myanmar (formerly Burma)",
    120: "Namibia",
    121: "Nauru",
    122: "Nepal",
    123: "Netherlands",
    124: "New Zealand",
    125: "Nicaragua",
    126: "Niger",
    127: "Nigeria",
    128: "North Macedonia",
    129: "Norway",
    130: "Oman",
    131: "Pakistan",
    132: "Palau",
    133: "Panama",
    134: "Papua New Guinea",
    135: "Paraguay",
    136: "Peru",
    137: "Philippines",
    138: "Poland",
    139: "Portugal",
    140: "Qatar",
    141: "Romania",
    142: "Russia",
    143: "Rwanda",
    144: "Saint Kitts and Nevis",
    145: "Saint Lucia",
    146: "Saint Vincent and the Grenadines",
    147: "Samoa",
    148: "San Marino",
    149: "Sao Tome and Principe",
    150: "Saudi Arabia",
    151: "Senegal",
    152: "Serbia",
    153: "Seychelles",
    154: "Sierra Leone",
    155: "Singapore",
    156: "Slovakia",
    157: "Slovenia",
    158: "Solomon Islands",
    159: "Somalia",
    160: "South Africa",
    161: "South Sudan",
    162: "Spain",
    163: "Sri Lanka",
    164: "Sudan",
    165: "Suriname",
    166: "Sweden",
    167: "Switzerland",
    168: "Syria",
    169: "Taiwan",
    170: "Tajikistan",
    171: "Tanzania",
    172: "Thailand",
    173: "Timor-Leste",
    174: "Togo",
    175: "Tonga",
    176: "Trinidad and Tobago",
    177: "Tunisia",
    178: "Turkey",
    179: "Turkmenistan",
    180: "Tuvalu",
    181: "Uganda",
    182: "Ukraine",
    183: "United Arab Emirates",
    184: "United Kingdom",
    185: "United States",
    186: "Uruguay",
    187: "Uzbekistan",
    188: "Vanuatu",
    189: "Vatican City",
    190: "Venezuela",
    191: "Vietnam",
    192: "Yemen",
    193: "Zambia",
    194: "Zimbabwe",
}

# Function to process the image and detect flags
def detect_flags(image_path, output_image_name):
    # Read the input image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img_rgb)  # Perform inference on the image

    # Extract the results
    predictions = results[0].boxes  # Extract bounding boxes
    labels = predictions.cls  # Class IDs
    scores = predictions.conf  # Confidence scores
    boxes = predictions.xywh  # Bounding boxes (x_center, y_center, width, height)

    # Draw the bounding boxes and labels on the image
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Only draw boxes with a high confidence score
            x_center, y_center, width, height = boxes[i]
            label = int(labels[i])  # Get the class ID
            class_name = flags.get(label, "Unknown")  # Get country name from the dictionary
            score = scores[i]

            # Convert box coordinates from relative to absolute values
            h, w, _ = img.shape
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Draw the rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{class_name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save the output image in the same directory with the given name
    output_path = f"{output_image_name}.jpg"
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")

# Example usage: Provide the flag image and the desired output image name
image_path = "indonesianflag2.jpg"  # The flag image located in the same directory
output_image_name = "output_indoflag"  # The desired name for the output image

detect_flags(image_path, output_image_name)
