import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from keras_tuner import RandomSearch  # Import Keras Tuner

# Constants
BATCH_SIZE = 32
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150  # Set image dimensions to 150x150
MODEL_FILE = 'lung_cancer_detection_model.keras'
EPOCHS = 10  # Default number of epochs for training

# Define dataset paths
base_data_dir = os.path.join(os.getcwd(), 'data')
train_data_dir = os.path.join(base_data_dir, "train")
val_data_dir = os.path.join(base_data_dir, "val")
test_data_dir = os.path.join(base_data_dir, "test")

# Create CNN model
def create_custom_cnn(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=1, hp=None):
    model = tf.keras.models.Sequential([
        layers.Input(shape=input_shape),
        Conv2D(hp.Int('conv1_filters', 32, 128, step=32), (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(hp.Int('conv2_filters', 64, 256, step=64), (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(hp.Int('conv3_filters', 128, 512, step=128), (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu'),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy', metrics=['accuracy'])
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

# Preprocess the image for prediction
def preprocess_image(img_path):
    try:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        new_image = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize to 150x150 for consistency
        processed_image = np.asarray(new_image) / 255.0
        img_array = np.expand_dims(processed_image, axis=0)
        return img_array
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# Generate Grad-CAM
def generate_gradcam(model, img_array):
    try:
        last_conv_layer = next(layer for layer in reversed(model.layers) if isinstance(layer, Conv2D))

        grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, last_conv_layer.output])

        with tf.GradientTape() as tape:
            model_output, conv_output = grad_model(img_array)
            class_id = tf.argmax(model_output[0])
            tape.watch(conv_output)
            grads = tape.gradient(model_output[:, class_id], conv_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]

        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= np.max(heatmap)  # Normalize

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

# Hyperparameter tuning
def tune_hyperparameters(train_generator, val_generator):
    tuner = RandomSearch(
        create_custom_cnn,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='lung_cancer_detection'
    )
    
    tuner.search(train_generator, epochs=EPOCHS, validation_data=val_generator)
    
    # Get the best model and parameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_model, best_hyperparameters

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
        print("No saved model found. Tuning hyperparameters and training a new model...")
        train_generator, val_generator = load_data(train_data_dir, val_data_dir)
        model, best_hyperparams = tune_hyperparameters(train_generator, val_generator)
        model.save(MODEL_FILE)  # Save the best model
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
