import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from PIL import Image
import gdown

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10
MODEL_FILE = '/content/drive/MyDrive/model_storage/lung_cancer_detection_model.h5'  # Updated path for saving the model

# Define the base data directory
base_data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_data_dir = os.path.join(base_data_dir, "train")
val_data_dir = os.path.join(base_data_dir, "val")
test_data_dir = os.path.join(base_data_dir, "test")

# Download model if not present
def download_model():
    # Ensure the directory exists
    model_dir = os.path.dirname(MODEL_FILE)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(MODEL_FILE):
        model_url = 'https://drive.google.com/uc?id=1lmzGa2wlcFfl8iU5sBgupKRbaIpKg_lL'
        gdown.download(model_url, MODEL_FILE, quiet=False)

# Download data if not present (replace with actual file IDs)
def download_data():
    data_files = {
        'train': 'FILE_ID_FOR_TRAIN',
        'val': 'FILE_ID_FOR_VAL',
        'test': 'FILE_ID_FOR_TEST'
    }

    for name, file_id in data_files.items():
        dir_path = os.path.join(base_data_dir, name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            file_url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(file_url, os.path.join(dir_path, f'{name}.zip'), quiet=False)

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

    plt.tight_layout()
    plt.show()

# Create DenseNet model
def create_densenet_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=1):
    base_model = DenseNet201(include_top=False, input_shape=input_shape, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)

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

# Generate Grad-CAM heatmap
def generate_gradcam(model, img_array):
    try:
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
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        return None

# Display Grad-CAM heatmap
def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range of 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use the "jet" colormap to colorize the heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Transform the heatmap into an image
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    
    # Resize the heatmap to match the image dimensions
    jet_heatmap = jet_heatmap.resize((img.shape[2], img.shape[1]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    plt.imshow(superimposed_img / 255.0)  # Normalize for display
    plt.axis('off')
    plt.show()

# Check if directory exists
def check_directory(path):
    if not os.path.exists(path):
        print(f"Directory does not exist: {path}")
        return False
    return True

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
    # Download model and data
    download_model()
    download_data()

    # Load model or create a new one if it doesn't exist
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

    # If model was not loaded, create and train a new one
    if model is None:
        print("No existing model found. Creating a new model...")
        model = create_densenet_model((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

            # Generate Grad-CAM
            heatmap = generate_gradcam(model, test_image_array)

            # Display Grad-CAM
            original_image = cv2.imread(test_image_path)
            original_image = cv2.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize to match model input
            display_gradcam(original_image, heatmap)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
