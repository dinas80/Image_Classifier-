# Image Classifier

This project aims to classify images into different categories using a convolutional neural network (CNN) implemented in TensorFlow.

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Saving](#model-saving)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

This project utilizes TensorFlow to build and train a CNN model for image classification. The dataset consists of images categorized into various classes. The trained model can predict the class of an input image with a certain level of accuracy.

## Usage

To use the image classifier:

1. Clone this repository:

git clone https://github.com/dinas80/Image_Classifier-.git

2. Install the required dependencies:

pip install -r requirements.txt


3. Prepare your dataset by organizing images into separate directories based on their classes.

4. Run the main script:

python run.py


This will load the dataset, preprocess it, train the model, evaluate its performance, and save the trained model.

## Dataset

The dataset should be organized into separate directories, each representing a class. The `load_and_preprocess_dataset` function loads the dataset from the specified directory, preprocesses it, and splits it into training, validation, and testing sets.

## Training

The `train_model` function builds and trains the CNN model using TensorFlow's Keras API. The model architecture consists of convolutional layers followed by max-pooling layers and fully connected layers. The training process is monitored using TensorBoard.

## Evaluation

The `evaluate_model` function evaluates the trained model's performance on the testing dataset, providing metrics such as loss and accuracy.

## Model Saving

After training, the trained model is saved in the `models` directory as `trained_model.h5`.

## Dependencies

- TensorFlow
- OpenCV
- Matplotlib
- NumPy

## License



