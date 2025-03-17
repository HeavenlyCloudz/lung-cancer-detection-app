import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Constants
BATCH_SIZE = 32
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
MODEL_FILE = 'lung_cancer_detection_model.keras'

# Define dataset paths
base_data_dir = os.path.join(os.getcwd(), 'data')
train_data_dir = os.path.join(base_data_dir, "train")
val_data_dir = os.path.join(base_data_dir, "val")
test_data_dir = os.path.join(base_data_dir, "test")

# Create CNN model
def create_custom_cnn(input_shape=(224, 224, 3), num_classes=1):
    model = tf.keras.models.Sequential([
        layers.Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load model from file
def load_model_file(model_file):
    if os.path.exists(model_file):
        try:
            model = tf.keras.models.load_model(model_file)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    return None

# Load data
def load_data(train_dir, val_dir):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, 
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, 
                                       horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE, class_mode='binary')

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE, class_mode='binary')

    return train_generator, val_generator

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Preprocess image for prediction
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.asarray(img_resized) / 255.0
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

# Generate Grad-CAM
def generate_gradcam(model, img_array):
    try:
        last_conv_layer = next(layer for layer in reversed(model.layers) if isinstance(layer, Conv2D))

        grad_model = tf.keras.models.Model(
            inputs=model.input, outputs=[model.output, last_conv_layer.output])

        with tf.GradientTape() as tape:
            model_output, conv_output = grad_model(img_array)
            class_id = tf.argmax(model_output[0])
            tape.watch(conv_output)
            grads = tape.gradient(model_output[:, class_id], conv_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]

        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Normalize

        return cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        return None

# Display Grad-CAM heatmap
def display_gradcam(img, heatmap, alpha=0.4):
    try:
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return superimposed_img
    except Exception as e:
        print(f"Error displaying Grad-CAM: {str(e)}")
        return None

# Train the model
def train_model(model, train_generator, val_generator):
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)
    model.save(MODEL_FILE)
    plot_training_history(history)

# Test the model
def test_model(model, test_data_dir):
    try:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE, class_mode='binary')

        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

# Main execution
if __name__ == "__main__":
    model = load_model_file(MODEL_FILE)

    if not model:
        print("No saved model found. Training a new model...")
        train_generator, val_generator = load_data(train_data_dir, val_data_dir)
        model = create_custom_cnn()
        train_model(model, train_generator, val_generator)
    else:
        print("Model loaded successfully.")

    test_model(model, test_data_dir)

    # Predict a single image with Grad-CAM
    test_image_path = input("Enter the path to the JPG/PNG test image: ")
    
    test_image_array = preprocess_image(test_image_path)
    if test_image_array is not None:
        predictions = model.predict(test_image_array)
        result = 'Cancerous' if predictions[0][0] > 0.5 else 'Non-Cancerous'
        print(f"Prediction: {result}")

        # Generate Grad-CAM
        heatmap = generate_gradcam(model, test_image_array)
        if heatmap is not None:
            original_image = cv2.imread(test_image_path)
            if original_image is not None:
                original_image = cv2.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                gradcam_result = display_gradcam(original_image, heatmap)

                # Display the Grad-CAM
                cv2.imshow("Grad-CAM", gradcam_result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Error: Could not read the input image.")
