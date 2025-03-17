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
last_conv_layer_name='conv2d_2')

# Create CNN model with explicit layer names
def create_custom_cnn(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=1):
    model = tf.keras.models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2d'),
        layers.MaxPooling2D((2, 2), name='max_pooling2d'),
        layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_1'),
        layers.MaxPooling2D((2, 2), name='max_pooling2d_1'),
        layers.Conv2D(256, (3, 3), activation='relu', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='max_pooling2d_2'),
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.Dense(128, activation='relu', name='dense_layer_1'),
        layers.Dense(num_classes, activation='sigmoid', name='output_layer')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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
        
        if img.mode != 'RGB':
            img = img.convert('RGB')  

        new_image = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  
        processed_image = np.asarray(new_image) / 255.0  

        if processed_image.ndim == 2:  
            processed_image = np.stack((processed_image,) * 3, axis=-1)

        img_array = np.expand_dims(processed_image, axis=0)  
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {str(e)}")
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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
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
        model = create_custom_cnn()  # Ensure model is created
        model, best_hyperparams = tune_hyperparameters(train_generator, val_generator)
        model.save(MODEL_FILE)  # Save the best model
    else:
        print("Model loaded successfully.")

    # Test with a dummy input to verify model output
    dummy_input = np.random.rand(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)  # Shape (1, 150, 150, 3)
    model(dummy_input)  # Check if this raises an error

    test_model(model, test_data_dir)

    # Predict a single image with Grad-CAM
    test_image_path = input("Enter the path to the JPG/PNG test image: ")
    
    test_image_array = preprocess_image(test_image_path)
    if test_image_array is not None:
        predictions = model.predict(test_image_array)
        result = 'Cancerous' if predictions[0][0] > 0.5 else 'Non-Cancerous'
        print(f"Prediction: {result}")

        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(test_image_array, model, last_conv_layer_name='conv2d_2')  # Adjust layer name as needed
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
