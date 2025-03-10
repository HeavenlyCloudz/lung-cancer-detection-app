import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Setting parameters of the image
image_height, image_width = 150, 150
batch_size = 32

# Load the model
model_file = os.path.abspath('lung_cancer_detection_model.h5')

# Check if the model exists
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    exit(1)  # Exit if the model is not found

# Load the model
try:
    model = tf.keras.models.load_model(model_file)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Data augmentation for training
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

# Validation data normalization
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the data
train_data_dir = os.path.abspath('data/train')
val_data_dir = os.path.abspath('data/val')

 # Load the data
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Create the CNN model
def create_cnn_model(input_shape=(150, 150, 3)):
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

# Train the model only if it does not exist
if model is None:
    # Defines the input shapes
    image_input_shape = (image_height, image_width, 3)

    # Create and compile the model
    model = create_cnn_model(image_input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=10
    )

    # Save the model
    model.save(model_file)

    # Plot training & validation accuracy and loss
    plot_training_history(history)

# Function to plot training history
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

    plt.tight_layout()
    plt.show()

# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap(model, img_array, class_index):
    last_conv_layer = model.layers[-1]  # Use the last Conv2D layer
    grad_model = Model(inputs=model.input, outputs=[model.output, last_conv_layer.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(np.expand_dims(img_array, axis=0))  # Ensure input shape is correct
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_output = conv_outputs[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.keras.activations.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) if tf.reduce_max(heatmap) > 0 else 1  # Normalize safely

    return heatmap.numpy()

def display_gradcam_heatmap(img_array, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to [0, 255]
    heatmap = cv2.resize(heatmap, (image_width, image_height))  # Resize the heatmap
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)  # Convert heatmap to RGB

    img_array = img_array * 255
    superimposed_img = cv2.addWeighted(img_array.astype(np.uint8), alpha, heatmap, 1 - alpha, 0)  # Overlay heatmap

    plt.imshow(superimposed_img)  # Show the image
    plt.axis('off')  # Turn off the axis
    plt.show()

# Function to load JPG/PNG images
def load_image(image_path):
    try:
        img_array = cv2.imread(image_path)
        img_array = cv2.resize(img_array, (image_width, image_height))  # Resize to match model input
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print("Error loading image:", e)
        return None

# Test the model and show Grad-CAM heatmap
if __name__ == "__main__":
    # Get user input for the image path
    test_image_path = input("Enter the path to the JPG/PNG test image to examine the image: ")

    # Load and process the image
    test_image_array = load_image(test_image_path)

    if test_image_array is not None:
        # Expand dimensions to add batch size
        test_image_array = np.expand_dims(test_image_array, axis=0)  # Shape should be (1, 150, 150, 3)

        # Get the predicted class
        predictions = model.predict(test_image_array)
        class_index = int(predictions[0] > 0.5)

        # Generate the Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(model, test_image_array[0], class_index)

        # Display the Grad-CAM heatmap
        display_gradcam_heatmap(test_image_array[0], heatmap)
