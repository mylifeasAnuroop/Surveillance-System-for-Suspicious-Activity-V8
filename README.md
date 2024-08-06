

# Surveillance System for Suspicious Activity and Weapon Detection

Author: Anuroop Arya

## Introduction

This project aims to enhance security in public spaces like schools by detecting suspicious activities and weapons using advanced computer vision techniques. The system utilizes a custom-trained YOLOv8 model, fine-tuned with data prepared using Roboflow. The application is built with Flask, allowing users to upload videos, images, or use a webcam for real-time detection.

## Project Overview

1. **Data Preparation**: Collected and annotated data using Roboflow, then exported the dataset for training.
2. **Model Training**: Fine-tuned the YOLOv8 model using the prepared dataset and evaluated model performance.
3. **Application Development**: Developed a Flask application for easy deployment, enabling users to upload videos, images, or use a webcam for detection.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Dependencies

Install the required dependencies using pip:

\`\`\`
bash
pip install ultralytics==8.0.196 roboflow Flask opencv-python Pillow

\`\`\`

## Usage

1. **Clone the repository**:

   \`\`\`bash
   git clone https://github.com/mylifeasAnuroop/Surveillance-System-for-Suspicious-Activity-V8.git
   cd Surveillance-System-for-Suspicious-Activity-V8
   \`\`\`

2. **Run the Flask application**:

   \`\`\`bash
   python app.py
   \`\`\`

3. **Access the application**:

   Open your web browser and go to \`http://127.0.0.1:5000\`.

4. **Upload an image or video, or use the webcam** to perform detection.

## Code Overview

### Flask Application

The Flask application handles routing for the web interface, allowing users to upload files and access the webcam for real-time detection.

The main routes in the Flask application are:

- **Home Page (\`/\`)**: Renders the home page (\`index.html\`).
- **Image Prediction (\`/predict_img\`)**: Handles image file uploads and performs object detection.
- **Video Prediction (\`/predict_video\`)**: Handles video file uploads and performs real-time object detection.
- **Webcam Feed (\`/webcam_feed\`)**: Provides a real-time webcam feed with object detection.

The complete Flask app code is located in the \`app.py\` file. You can view and modify it as needed.

### YOLOv8 Model

The YOLOv8 model is imported and used in the Flask routes for detection, fine-tuned with the weights saved in \`best.pt\`.

### Data Preparation with Roboflow

Collected and annotated data using Roboflow, then downloaded the dataset for YOLOv8 training.

## Example Code Snippets

### YOLOv8 Model Integration

\`\`\`python
from ultralytics import YOLO

model = YOLO('path/to/best.pt')

# Example of running detection on an image
results = model('path/to/image.jpg')
\`\`\`

### Data Preparation with Roboflow

\`\`\`python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("your-project-name")
dataset = project.version(1).download("yolov8")
\`\`\`

## Conclusion

This surveillance system leverages deep learning and computer vision to provide a robust solution for detecting suspicious activities and weapons in public spaces. With its easy-to-use interface and real-time capabilities, it can be an effective tool for enhancing security and safety.
