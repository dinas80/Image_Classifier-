import tensorflow as tf
import cv2
import numpy as np

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize the image to match the input size of the model
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return img

def classify_image(model, image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Predict the class probabilities
    predictions = model.predict(img)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions[0])

    class_names = {
        0: 'Accessories',
        1: 'Artifacts',
        2: 'Bags',
        3: 'Beauty',
        4: 'Books',
        5: 'Fashion',
        6: 'Games',
        7: 'Home',
        8: 'Nutrition',
        9: 'Pet Products',
        10: 'Sports',
        11: 'Stationary'
    }
    
    # Print the name of the predicted class
    predicted_class_name = class_names.get(predicted_class, 'Unknown')
    
    return predicted_class_name

def main():
    # Load the trained model
    model_path = '/home/dodzz/repos/Image_Classifier-/models/trained_model.h5'
    model = load_trained_model(model_path)
    
    # Path to the test image
    test_image_path = '/home/dodzz/repos/Image_Classifier-/test/1.jpg'
    
    # Classify the test image
    predicted_class = classify_image(model, test_image_path)
    
    # Print the predicted class
    print(f'Predicted class : {predicted_class}')

if __name__ == "__main__":
    main()
