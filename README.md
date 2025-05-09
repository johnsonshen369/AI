# AI

# 🏀 Sport Ball Detection - YOLOv5 Project

This project uses **YOLOv5** to detect basketballs in videos. It is built with PyTorch and includes custom training for real-world game footage.

## 🚀 Features

- Detects basketballs in images and videos
- Trained on custom dataset from game scenes
- Supports video input, outputs detection results
- Lightweight and easy to run locally

## 🖼️ Sample Detection Image

![Detection Example](sport_ball_detect/yolov5/output_with_balls.png)

## 🎥 Demo Video

Here is the demo video showcasing the detection in action:

[Watch the demo video](sport_ball_detect/yolov5/output_with_balls.mp4)

> Tip: You can also upload the video to the "Releases" section or use an external host (e.g., YouTube, Streamable) for better accessibility.

## 🔧 How to Use

# bash
# Clone the repo
git clone https://github.com/johnsonshen369/AI.git
cd AI/sport_ball_detect

# Install dependencies
pip install -r requirements.txt

# Run detection
python detect.py --weights models/yolov5x6.pt --source clip.mp4


#Project Structure
sport_ball_detect/
├── detect.py               # Main script to run detection
├── models/                 # Pre-trained models
│   └── yolov5x6.pt         # YOLOv5 weights for detection
├── yolov5/                 # YOLOv5 source code and other resources
│   ├── output_with_balls.png  # Example detection output
│   └── output_with_balls.mp4  # Example input/output video
├── images/                   # Dataset folder (images & labels)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

#Model Training
1.To train your own YOLOv5 model on basketball data:
Prepare your dataset: Organize your images and labels in the following format:
├── images/
│   ├── train/        # Training images
│   └── val/          # Validation images
├── labels/
│   ├── train/        # Training labels
│   └── val/          # Validation labels

2.Train the model using YOLOv5:
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt

3.Monitor the training with TensorBoard (optional):
tensorboard --logdir=runs

#Contributing
We welcome contributions to improve the project! Here's how you can help:
Fork the repository and create a new branch (git checkout -b feature-name).
Make your changes.
Commit your changes (git commit -am 'Add new feature').
Push to your branch (git push origin feature-name).
Create a Pull Request describing your changes.
Feel free to suggest new features or improvements. All contributions are appreciated!

#Requirements
This project depends on the following libraries:
Python 3.x
PyTorch
OpenCV
NumPy
Matplotlib
YOLOv5 repository (for model architecture)
To install the dependencies:

##Acknowledgments
YOLOv5 for the detection model: https://github.com/ultralytics/yolov5
OpenCV for image and video processing
PyTorch for model training and deployment
Dataset contributors for providing game footage and labels

##License
This project is licensed under the MIT License - see the LICENSE file for details.
