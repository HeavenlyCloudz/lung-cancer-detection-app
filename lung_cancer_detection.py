import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm
import cv2

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
MODEL_FILE = 'lung_cancer_detection_model.keras'
BATCH_SIZE = 32

# Define dataset paths
base_data_dir = os.path.join(os.getcwd(), 'data')
train_data_dir = os.path.join(base_data_dir, "train")
val_data_dir = os.path.join(base_data_dir, "val")
test_data_dir = os.path.join(base_data_dir, "test")

# Create DenseNet model
def create_densenet_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=1):
    densenet_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in densenet_model.layers:
        layer.trainable = False
    x = densenet_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=densenet_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load model from file
def load_model_file(model_file):
    if os.path.exists(model_file):
        try:
            model = load_model(model_file)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    else:
        print("No saved model found.")
    return None

# Load data
def load_data(train_dir, val_dir):
    try:
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                           width_shift_range=0.2, height_shift_range=0.2,
                                           shear_range=0.2, zoom_range=0.2,
                                           horizontal_flip=True, fill_mode='nearest')

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE, class_mode='binary')

        val_generator = val_datagen.flow_from_directory(
            val_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE, class_mode='binary')

        return train_generator, val_generator
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

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
        img = Image.open(img_path).convert('RGB')
        new_image = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize to 224x224
        processed_image = np.asarray(new_image) / 255.0
        
        if processed_image.ndim == 2:
            processed_image = np.stack((processed_image,) * 3, axis=-1)

        img_array = np.expand_dims(processed_image, axis=0)  # Shape becomes (1, 224, 224, 3)
        print(f"Processed image shape: {img_array.shape}")  # Debug output
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# Find the last convolutional layer name
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if 'conv' in layer.name:  # Check if the layer is a convolutional layer
            return layer.name
    return None

# Generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap /= tf.reduce_max(heatmap)  # Normalize
    
    return heatmap.numpy()

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
    # Calculate class weights
    y_train = train_generator.classes
    class_labels = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=class_labels, y=y_train)
    class_weights = {i: weights[i] for i in range(len(class_labels))}
    
    history = model.fit(train_generator, validation_data=val_generator, epochs=10, class_weight=class_weights)
    model.save(MODEL_FILE)
    plot_training_history(history)

# Test the model
def test_model(model, test_data_dir):
    try:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), 
            batch_size=BATCH_SIZE, class_mode='binary')

        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Additional metrics
        y_pred = model.predict(test_generator)
        y_pred_classes = np.where(y_pred > 0.5, 1, 0)
        cm = confusion_matrix(test_generator.classes, y_pred_classes)

        # Calculate and print precision and recall
        tp = cm[1, 1]  # True Positives
        fp = cm[0, 1]  # False Positives
        fn = cm[1, 0]  # False Negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

    except Exception as e:
        print(f"Error during testing: {str(e)}")

# Main execution
if __name__ == "__main__":
    model = load_model_file(MODEL_FILE)

    if not model:
        print("No saved model found. Training a new model...")
        train_generator, val_generator = load_data(train_data_dir, val_data_dir)
        model = create_densenet_model()  # Create DenseNet model
        train_model(model, train_generator, val_generator)  # Train the model
    else:
        print("Model loaded successfully.")

    # Get the last convolutional layer name
    last_conv_layer_name = get_last_conv_layer_name(model)

    # Test the model
    test_model(model, test_data_dir)

    # Predict a single image with Grad-CAM
    test_image_path = input("Enter the path to the JPG/PNG test image: ")
    
    test_image_array = preprocess_image(test_image_path)
    if test_image_array is not None:
        predictions = model.predict(test_image_array)
        result = 'Cancerous' if predictions[0][0] > 0.5 else 'Non-Cancerous'
        print(f"Prediction: {result}")

        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(test_image_array, model, last_conv_layer_name)  # Use the last conv layer
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
