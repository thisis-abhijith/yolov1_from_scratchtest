# YOLO v1 from Scratch
![image](https://github.com/user-attachments/assets/5402d69b-bcfa-4b02-bf98-0cc002c5a446)

# YOLO v1 from Scratch

## Table of Contents
- Overview
- Requirements
- Installation
- Dataset
- Training
- Evaluation
- Results
- References

## Overview
YOLO v1 revolutionized object detection by processing images in a single pass, rather than a region proposal-based approach like Fast R-CNN. This implementation aims to understand the YOLO v1 architecture and build it from scratch. Key features include:
- Single neural network architecture for object detection.
- Predicts bounding boxes and class probabilities for multiple objects in a single forward pass.
- Optimized for real-time object detection with high speed and decent accuracy.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- tqdm (for progress bars)

You can install the required packages using:

```bash
pip install -r requirements.txt
```
Installation
Clone the repository and install dependencies:

```bash

git clone https://github.com/your-username/yolo-v1-from-scratch.git
cd yolo-v1-from-scratch
pip install -r requirements.txt
```
Dataset
To train the model, you can use the PASCAL VOC dataset or any custom dataset in the COCO format. Download the PASCAL VOC dataset using:

```bash

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar`
```
Make sure to preprocess the dataset by resizing and normalizing the images as required by YOLO.\

Training:
To start training the YOLO v1 model, run the following command:

```bash

python train.py --config config.json
```
The config.json file contains all the hyperparameters, paths, and settings required to train the model, including batch size, learning rate, and dataset path.

Evaluation
To evaluate the model on the validation/test set, use:

```
python evaluate.py --weights path_to_weights.pth --data path_to_data
```

Results
After training the model, you can visualize the predictions on an input image by running:

```
python test.py --image path_to_image.jpg --weights path_to_weights.pth
```

## References
- [YOLO v1 Paper: You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)

## Need Help?

If you have any doubts or questions, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/manan-chandna-697588257/).




