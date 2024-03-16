import tensorflow as tf

# Load the dataset and get class names
data_dir = 'data'
data = tf.keras.utils.image_dataset_from_directory(data_dir, seed=123)
class_names = data.class_names

# Print class labels and their corresponding names
for label, name in enumerate(class_names):
    print(f"Label {label}: {name}")
