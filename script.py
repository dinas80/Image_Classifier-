import tensorflow as tf
import os 
import cv2 
import imghdr
import numpy as np
from matplotlib import pyplot as plt

def load_and_preprocess_dataset(data_dir):
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    
    # Remove dodgy images 
    for image_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, image_class)
        if os.path.isdir(class_dir):
            for image in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image)
                try:
                    img = cv2.imread(image_path)
                    tip = imghdr.what(image_path)
                    if tip not in image_exts:
                        print('Image not in ext list {}'.format(image_path))
                        os.remove(image_path)
                except Exception as e:
                    print('Issue with image {}'.format(image_path))
    
    # Build dataset
    data = tf.keras.utils.image_dataset_from_directory(data_dir, seed=123, label_mode='categorical', image_size=(256, 256), batch_size=32)
    class_names = data.class_names
    data = data.map(lambda x, y: (x / 255.0, y))  # Normalization
    
    # Get the number of classes
    num_classes = len(class_names)
    
    # Split dataset
    data_len = len(data)
    train_size = int(data_len * 0.7)
    val_size = int(data_len * 0.2)
    test_size = data_len - train_size - val_size
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    
    return train, val, test, num_classes

def train_model(train_data, val_data, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Use softmax for multiclass classification
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use categorical crossentropy for multiclass classification
                  metrics=['accuracy'])

    # Train the model
    logdir = 'logs'
    tensboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensboard_callback])
    
    return model, hist

def evaluate_model(model, test_data):
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Main script
data_directories = ['data']  # List of directories containing datasets
for data_dir in data_directories:
    train_data, val_data, test_data, num_classes = load_and_preprocess_dataset(data_dir)
    trained_model, history = train_model(train_data, val_data, num_classes)
    evaluate_model(trained_model, test_data)
