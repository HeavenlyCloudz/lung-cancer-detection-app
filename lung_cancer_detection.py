import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10
MODEL_FILE = os.path.abspath('lung_cancer_detection_model.h5')

# Define the base data directory
base_data_dir = os.path.join(os.path.dirname(__file__), 'data')
test_data_dir = os.path.join(base_data_dir, 'test')

def plot_training_history(history):
    """Plot the training and validation accuracy and loss."""
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

    plt.tight_layout()
    plt.show()

def create_cnn_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)):
    """Create a CNN model."""
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def load_model_file(model_file):
    """Load the model from the specified file."""
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

def load_data(train_dir, val_dir):
    """Load training and validation data from directories."""
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

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

    # Check if the generator has enough data
    if train_generator.samples < BATCH_SIZE or val_generator.samples < BATCH_SIZE:
        raise ValueError("Not enough data in training or validation set for the specified batch size.")

    return train_generator, val_generator

def preprocess_image(image_path):
    """Load and preprocess an image from a local file path."""
    img = Image.open(image_path)

    if img.mode == 'RGBA':
        img = img.convert('RGB')

    new_image = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    processed_image = np.asarray(new_image) / 255.0  # Normalize directly

    image = np.expand_dims(processed_image, axis=0)
    return image

def load_and_preprocess_images_from_folder(folder_path):
    """Load and preprocess all images from a specified folder."""
    processed_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            processed_image = preprocess_image(image_path)
            processed_images.append(processed_image)
    return np.vstack(processed_images) if processed_images else None

def generate_gradcam(model, img_array):
    """Generate Grad-CAM heatmap."""
    last_conv_layer = model.get_layer('conv2d_2')
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, last_conv_layer.output])

    with tf.GradientTape() as tape:
        model_output, last_conv_layer_output = grad_model(img_array)
        class_id = tf.argmax(model_output[0])
        grads = tape.gradient(model_output[:, class_id], last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))
    return heatmap

def check_directory(path):
    """Check if the directory exists."""
    if not os.path.exists(path):
        print(f"Directory does not exist: {path}")
        return False
    return True

def test_model(model, test_data_dir, epochs=1):
    """Load test data and evaluate the model over a number of epochs."""
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

if __name__ == "__main__":
    train_data_dir = os.path.join(base_data_dir, "train")
    val_data_dir = os.path.join(base_data_dir, "val")

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

    # Compile and evaluate the model if it is loaded successfully
    if model is not None:
        model.summary()

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(val_generator)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Test the model
        if check_directory(test_data_dir):
            # Ask user for the number of epochs to test
            test_epochs = int(input("Enter the number of epochs for testing (default is 1): ") or 1)
            test_model(model, test_data_dir, epochs=test_epochs)
        else:
            print(f"Test data directory does not exist: {test_data_dir}")

    # If model is not loaded, create and train a new one
    if model is None:
        model = create_cnn_model((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Calculate steps per epoch
        steps_per_epoch = train_generator.samples // BATCH_SIZE
        validation_steps = val_generator.samples // BATCH_SIZE

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=EPOCHS
        )

        model.save(MODEL_FILE)
        plot_training_history(history)

    # Load and predict on all images in the base data directory
    try:
        all_processed_images = load_and_preprocess_images_from_folder(base_data_dir)
        if all_processed_images is not None:
            print("Processed images shape:", all_processed_images.shape)

            if model:
                predictions = model.predict(all_processed_images)
                for i, prediction in enumerate(predictions):
                    result = 'Cancerous' if prediction[0] > 0.5 else 'Non-Cancerous'
                    print(f"Image {i + 1}: The model predicts the image is: {result}")
    except Exception as e:
        print(f"Error loading images: {str(e)}")

    test_image_path = input("Enter the path to the JPG/PNG test image: ")
    
    try:
        test_image_array = preprocess_image(test_image_path)

        if test_image_array is not None and model:
            predictions = model.predict(test_image_array)
            class_index = int(predictions[0] > 0.5)

            heatmap = generate_gradcam(model, test_image_array)
            plt.axis('off')
            plt.matshow(heatmap)
            plt.show()

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
