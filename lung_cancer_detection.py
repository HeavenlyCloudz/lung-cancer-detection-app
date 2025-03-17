import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from PIL import Image

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10

# Model Path
MODEL_FILE = 'lung_cancer_detection_model.keras'

# Define the base data directory
base_data_dir = os.path.join(os.getcwd(), 'data')
train_data_dir = os.path.join(base_data_dir, "train")
val_data_dir = os.path.join(base_data_dir, "val")
test_data_dir = os.path.join(base_data_dir, "test")

# Create Custom CNN model
def create_custom_cnn(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=1):
    model = tf.keras.models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))  # Use 'softmax' for multi-class
    return model

# Load model from file
def load_model_file(model_file):
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_file)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Load training and validation data
def load_data(train_dir, val_dir):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        if train_generator.samples < BATCH_SIZE or val_generator.samples < BATCH_SIZE:
            raise ValueError("Not enough data in training or validation set for the specified batch size.")

        return train_generator, val_generator
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Preprocess image for prediction
def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        new_image = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        processed_image = np.asarray(new_image) / 255.0
        return np.expand_dims(processed_image, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

# Check if directory exists
def check_directory(path):
    if not os.path.exists(path):
        print(f"Directory does not exist: {path}")
        return False
    return True

# Generate Grad-CAM heatmap
def generate_gradcam(model, img_array):
    try:
        last_conv_layer = model.get_layer(index=5)  # Update index based on your model's layers
        grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, last_conv_layer.output])

        with tf.GradientTape() as tape:
            model_output, last_conv_layer_output = grad_model(img_array)
            class_id = tf.argmax(model_output[0])
            grads = tape.gradient(model_output[:, class_id], last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        last_conv_layer_output = last_conv_layer_output[0]

        # Calculate the heatmap
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0)  # ReLU
        heatmap /= tf.reduce_max(heatmap)  # Normalize

        heatmap = cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))
        return heatmap
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        return None

# Display Grad-CAM heatmap
def display_gradcam(img, heatmap, alpha=0.4):
    try:
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))  # Resize to match original image dimensions
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

        # Overlay the heatmap on the original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)  # Clip values to valid range
        return superimposed_img
    except Exception as e:
        print(f"Error displaying Grad-CAM: {str(e)}")
        return None

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()

# Test the model
def test_model(model, test_data_dir, epochs=1):
    try:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
        
        for epoch in range(epochs):
            test_loss, test_accuracy = model.evaluate(test_generator)
            print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Load model
    model = load_model_file(MODEL_FILE)

    # Verify dataset paths
    if not check_directory(base_data_dir) or not check_directory(train_data_dir) or not check_directory(val_data_dir):
        exit(1)

    # Load training data
    try:
        train_generator, val_generator = load_data(train_data_dir, val_data_dir)
    except ValueError as ve:
        print(ve)
        exit(1)

    # If model was not loaded, inform the user
    if model is None:
        print("No existing model found. You can train a new model if desired.")
    else:
        print("Loaded existing model.")

    # Load and predict on all images in the test directory
    try:
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            test_data_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        for i in range(test_generator.samples):
            img_array = test_generator[i][0]  # Get the image array
            predictions = model.predict(img_array)
            result = 'Cancerous' if predictions[0][0] > 0.5 else 'Non-Cancerous'
            print(f"Image {i + 1}: The model predicts the image is: {result}")
    except Exception as e:
        print(f"Error loading images: {str(e)}")

    test_image_path = input("Enter the path to the JPG/PNG test image: ")
    
    try:
        test_image_array = preprocess_image(test_image_path)

        if test_image_array is not None and model:
            predictions = model.predict(test_image_array)
            class_index = int(predictions[0] > 0.5)

            # Generate Grad-CAM
            heatmap = generate_gradcam(model, test_image_array)

            # Display Grad-CAM
            original_image = cv2.imread(test_image_path)
            original_image = cv2.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize to match model input
            superimposed_img = display_gradcam(original_image, heatmap)

            # Show the result
            cv2.imshow("Grad-CAM", superimposed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
