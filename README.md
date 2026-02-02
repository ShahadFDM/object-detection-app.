# object-detection-app.
A real-time object detection application running entirely in the browser using TensorFlow.js.

## Features

- **Multiple Detection Models**:
  -  **COCO-SSD Lite**: Fast detection for 80 object classes.
  -  **COCO-SSD Full**: High-accuracy detection.
  -  **MoveNet**: Real-time human pose estimation (17 keypoints).
  -  **HandPose**: Hand landmark detection (21 points per hand).
- **Scene Description**: Generates natural language descriptions of the scene based on detections.
- **Smart Analytics**: Real-time counting, FPS tracking, and confidence scoring.
- **Privacy Focused**: All processing happens locally in your browser. No images are sent to any server.

## Technologies

- HTML5 / CSS3
- JavaScript (ES6+)
- [TensorFlow.js](https://www.tensorflow.org/js)
- Pre-trained Models: COCO-SSD, MoveNet, HandPose
