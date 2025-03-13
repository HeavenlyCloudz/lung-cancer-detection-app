import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10
MODEL_FILE = os.path.abspath('lung_cancer_detection_model.h5')

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
    """Create and return a CNN model."""
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    return model

def load_model_file(model_file):
    """Load the model from the specified file."""
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None

    try:
        model = tf.keras.models.load_model(model_file)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_data(train_dir, val_dir):
    """Load training and validation data from directories."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

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

    return train_generator, val_generator

def generate_gradcam_heatmap(model, img_array, class_index):
    """Generate a Grad-CAM heatmap for a given image array."""
    last_conv_layer = model.layers[-4]  # Use the last Conv2D layer
    grad_model = Model(inputs=model.input, outputs=[model.output, last_conv_layer.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(np.expand_dims(img_array, axis=0))
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_output = conv_outputs[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.keras.activations.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) if tf.reduce_max(heatmap) > 0 else 1

    return heatmap.numpy()

def display_gradcam_heatmap(img_array, heatmap, alpha=0.4):
    """Display the Grad-CAM heatmap overlayed on the original image."""
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (IMAGE_WIDTH, IMAGE_HEIGHT))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)

    img_array = img_array * 255
    superimposed_img = cv2.addWeighted(img_array.astype(np.uint8), alpha, heatmap, 1 - alpha, 0)

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

def load_image(image_path):
    """Load and preprocess an image from a given path."""
    try:
        img_array = cv2.imread(image_path)
        img_array = cv2.resize(img_array, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print("Error loading image:", e)
        return None

if __name__ == "__main__":
    # Load the model
    model = load_model_file(MODEL_FILE)

    # Load data
    train_data_dir = 'C:\\Users\\Antoru Grace Inc\\.vscode\\CNN\\streamlit_project\\data\\train'
    val_data_dir = 'C:\\Users\\Antoru Grace Inc\\.vscode\\CNN\\streamlit_project\\data\\val'
    
    # Check if directories exist
    if not os.path.exists(train_data_dir):
        print(f"Training data directory does not exist: {train_data_dir}")
        exit(1)
    if not os.path.exists(val_data_dir):
        print(f"Validation data directory does not exist: {val_data_dir}")
        exit(1)

    train_generator, val_generator = load_data(train_data_dir, val_data_dir)

    # Train the model if it does not exist
    if model is None:
        image_input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        model = create_cnn_model(image_input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE,
            epochs=EPOCHS
        )

        model.save(MODEL_FILE)
        plot_training_history(history)

    # Test the model and show Grad-CAM heatmap
    test_image_path = input("Enter the path to the JPG/PNG test image: ")
    test_image_array = load_image(test_image_path)

    if test_image_array is not None:
        test_image_array = np.expand_dims(test_image_array, axis=0)
        predictions = model.predict(test_image_array)
        class_index = int(predictions[0] > 0.5)

        heatmap = generate_gradcam_heatmap(model, test_image_array[0], class_index)
        display_gradcam_heatmap(test_image_array[0], heatmap)
