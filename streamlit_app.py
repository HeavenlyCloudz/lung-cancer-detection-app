import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image  # Importing PIL for image processing

# Page configurations
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="C:\Users\Antoru Grace Inc\Downloads\logo"
    layout="wide" # Layout is wide
)

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'lung_cancer_detection_model.h5')

# Set dataset paths using relative paths
base_data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_data_dir = os.path.join(base_data_dir, 'train')
val_data_dir = os.path.join(base_data_dir, 'val')

# Load the model
model = None
val_loss, val_accuracy = None, None

try:
    model = load_model(MODEL_FILE)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model

    # Evaluate the model using validation data
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_data_dir, 
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=32, 
        class_mode='binary'
    )

    # This step builds the compiled metrics
    val_loss, val_accuracy = model.evaluate(val_generator)
except Exception as e:
    model = None
    st.error(f"Error loading model: {str(e)}")

# Preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path)

    # Convert to RGB if the image has an alpha channel
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    new_image = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    processed_image = np.asarray(new_image) / 255.0  # Normalize the image
    img_array = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    return img_array

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
    # Access the last convolutional layer
    last_conv_layer = model.get_layer('conv2d_2')  # Use the correct layer name
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, last_conv_layer.output])

    with tf.GradientTape() as tape:
        model_output, last_conv_layer_output = grad_model(img_array)  # Ensure img_array is of shape (1, 150, 150, 3)
        class_id = tf.argmax(model_output[0])  # Get the index of the highest probability
        grads = tape.gradient(model_output[:, class_id], last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    last_conv_layer_output = last_conv_layer_output[0]  # Use the first image in the batch

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)  # Normalize the heatmap
    heatmap = cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))
    return heatmap

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
    plt.savefig('training_history.png')
    plt.close()

# Creates CNN model
def create_cnn_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Use Input layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Function to train the model
def train_model(epochs, batch_size, use_early_stopping):
    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Check if paths exist
    if not os.path.exists(train_data_dir):
        st.error(f"Training data path does not exist: {train_data_dir}")
        return
    if not os.path.exists(val_data_dir):
        st.error(f"Validation data path does not exist: {val_data_dir}")
        return

    # Load data
    try:
        train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                            batch_size=batch_size, class_mode='binary')
        val_generator = val_datagen.flow_from_directory(val_data_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        batch_size=batch_size, class_mode='binary')
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Create and compile the model
    model = create_cnn_model((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Prepare callbacks
    callbacks = []
    if use_early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks.append(early_stopping)

    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = val_generator.samples // batch_size

    # Train the model
    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch,
                        validation_data=val_generator, validation_steps=validation_steps,
                        epochs=epochs, callbacks=callbacks)

    # Save the model
    model.save(MODEL_FILE)

    # Plot and save the training history
    plot_training_history(history)

# Streamlit UI
st.title("Lung Cancer Detection")
st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6; /* Light blue color */
    }
    .section {
        background-image: url('https://jnj-content-lab2.brightspotcdn.com/dims4/default/78c6313/2147483647/strip/false/crop/1440x666+0+0/resize/1440x666!/quality/90/?url=https%3A%2F%2Fjnj-production-jnj.s3.us-east-1.amazonaws.com%2Fbrightspot%2F1b%2F32%2F2e138abbf1792e49103c9e3516a8%2Fno-one-would-believe-me-when-i-suspected-i-had-lung-cancer-0923-new.jpg');
        background-size: cover; /* Ensure the image covers the section */
        background-repeat: no-repeat;
        background-position: center;
        padding: 60px; /* Increased padding for a bigger section */
        border-radius: 10px;
        color: black; /* Text color */
        margin: 20px 0;
        height: 400px; /* Increased height for the section */
    }
    
    .sidebar .sidebar-content {
        background-color: #ADD8E6; /* Light blue color */
        color: black; /* Change text color for visibility */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("Thank you for using ONCOSCAN")
st.write("CNNs are the preferred network for detecting lung cancer due to their ability to process image data. They can perform tasks such as classification, segmentation, and object recognition. In the case of lung cancer detection, CNNs have surpassed radiologists.")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Controls")

# Hyperparameter inputs
epochs = st.sidebar.number_input("Number of epochs", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=32)
use_early_stopping = st.sidebar.checkbox("Use Early Stopping", value=True)  # Checkbox for early stopping

# Button to train model
if st.sidebar.button("Train Model"):
    with st.spinner("Training the model..."):
        train_model(epochs, batch_size, use_early_stopping)  # Pass early stopping choice
    st.success("Model training complete!")  # This will display after training is done

# Display optimizer details and evaluation metrics in the sidebar
if model:
    st.sidebar.subheader("Optimizer Details")
    optimizer = model.optimizer
    optimizer_details = {
        "Optimizer": optimizer.__class__.__name__,
        "Learning Rate": optimizer.learning_rate.numpy()
    }
    for key, value in optimizer_details.items():
        st.sidebar.write(f"{key}: {value}")

    # Show validation loss and accuracy after evaluation
    if val_loss is not None and val_accuracy is not None:
        st.sidebar.subheader("Validation Metrics")
        st.sidebar.write(f"Validation Loss: {val_loss:.4f}")
        st.sidebar.write(f"Validation Accuracy: {val_accuracy:.4f}")

# Display training history if it exists
if os.path.exists('training_history.png'):
    st.subheader("Training History")
    st.image('training_history.png', caption='Training History', use_container_width=True)

# Display model summary at the bottom of the page
if model:
    st.subheader("Model Summary")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text('\n'.join(model_summary))

# Image upload for prediction
uploaded_file = st.sidebar.file_uploader("Upload your image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    img_array = preprocess_image("temp_image.jpg")  # Ensure shape is (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    if model:  # Ensure model is loaded
        try:
            prediction = model.predict(img_array)  # Predict on the processed image
            result = 'Cancerous' if prediction[0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result:")
            st.write(f"The model predicts the image is: **{result}**")

            # Generate and display Grad-CAM heatmap
            heatmap = generate_gradcam(model, img_array)  # Pass the preprocessed image
            st.image("temp_image.jpg", caption='Uploaded Image', use_container_width=True)

            # Display heatmap as an image
            plt.imshow(heatmap, cmap='jet')
            plt.axis('off')
            plt.colorbar()
            plt.savefig('gradcam.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            st.image('gradcam.png', caption='Grad-CAM', use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    os.remove("temp_image.jpg")

# Mobile Upload Option
st.sidebar.header("Take a Picture")
photo = st.sidebar.file_uploader("Capture a photo", type=["jpg", "jpeg", "png"])

if photo is not None:
    with open("captured_image.jpg", "wb") as f:
        f.write(photo.getbuffer())

    img_array = preprocess_image("captured_image.jpg")  # Ensure shape is (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    if model:  # Ensure model is loaded
        try:
            prediction = model.predict(img_array)  # Predict on the processed image
            result = 'Cancerous' if prediction[0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result for Captured Image:")
            st.write(f"The model predicts the image is: **{result}**")

            # Generate and display Grad-CAM heatmap
            heatmap = generate_gradcam(model, img_array)  # Pass the preprocessed image
            st.image("captured_image.jpg", caption='Captured Image', use_container_width=True)

            # Display heatmap as an image
            plt.imshow(heatmap, cmap='jet')
            plt.axis('off')
            plt.colorbar()
            plt.savefig('gradcam_captured.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            st.image('gradcam_captured.png', caption='Grad-CAM for Captured Image', use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    os.remove("captured_image.jpg")
