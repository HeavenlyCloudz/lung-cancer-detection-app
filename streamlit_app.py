import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150
BATCH_SIZE = 32

# Set paths for saving the model and data
MODEL_FILE = 'lung_cancer_detection_model.keras'
base_data_dir = os.path.join(os.getcwd(), 'data')
train_data_dir = os.path.join(base_data_dir, 'train')
val_data_dir = os.path.join(base_data_dir, 'val')
test_data_dir = os.path.join(base_data_dir, 'test')

# Function to create Custom CNN model (for reference)
def create_custom_cnn(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=1):
    model = tf.keras.models.Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=input_shape))
    
    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output
    model.add(layers.Flatten())
    
    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='sigmoid'))  # Use 'softmax' for multi-class

    return model

# Load model from file
def load_model_file():
    if os.path.exists(MODEL_FILE):
        try:
            model = load_model(MODEL_FILE)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.warning("No pre-trained model found.")
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

        return train_generator, val_generator
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Preprocess the image for prediction
def preprocess_image(img_path):
    try:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        new_image = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        processed_image = np.asarray(new_image) / 255.0
        img_array = np.expand_dims(processed_image, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Generate Grad-CAM heatmap
def generate_gradcam(model, img_array):
    try:
        last_conv_layer = model.get_layer(index=-3)  # Get the last Conv2D layer
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
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None

# Display Grad-CAM heatmap
def display_gradcam(img, heatmap, alpha=0.4):
    try:
        heatmap = np.uint8(255 * heatmap)
        
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[2], img.shape[1]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
        
        superimposed_img = jet_heatmap * alpha + img
        return superimposed_img
    except Exception as e:
        st.error(f"Error displaying Grad-CAM: {str(e)}")
        return None

# Function to plot training history
def plot_training_history(history):
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('Model Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_title('Model Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting training history: {str(e)}")

# Function to test the model
def test_model(model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    try:
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        test_loss, test_accuracy = model.evaluate(test_generator)
        st.sidebar.write(f"Test Loss: {test_loss:.4f}")
        st.sidebar.write(f"Test Accuracy: {test_accuracy:.4f}")

        y_pred = model.predict(test_generator)
        y_pred_classes = np.where(y_pred > 0.5, 1, 0)
        cm = confusion_matrix(test_generator.classes, y_pred_classes)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=['Non-Cancerous', 'Cancerous'], 
                     yticklabels=['Non-Cancerous', 'Cancerous'], ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during testing: {str(e)}")

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
        background-size: cover; 
        background-repeat: no-repeat;
        background-position: center;
        padding: 60px; 
        border-radius: 10px;
        color: black; 
        margin: 20px 0;
        height: 400px; 
    }
    
    .sidebar .sidebar-content {
        background-color: #ADD8E6; 
        color: black; 
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

# Load the model
model = load_model_file()

# Hyperparameter inputs
epochs = st.sidebar.number_input("Number of epochs", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=32)

# Button to train model
if st.sidebar.button("Train Model"):
    with st.spinner("Training the model..."):
        train_generator, val_generator = load_data(train_data_dir, val_data_dir)
        if train_generator is not None and val_generator is not None:
            steps_per_epoch = train_generator.samples // batch_size
            validation_steps = val_generator.samples // batch_size

            history = model.fit(train_generator, steps_per_epoch=steps_per_epoch,
                                validation_data=val_generator, validation_steps=validation_steps,
                                epochs=epochs)

            model.save(MODEL_FILE)
            st.success("Model trained and saved successfully!")
            plot_training_history(history)

# Button to test model
if st.sidebar.button("Test Model"):
    if model:
        with st.spinner("Testing the model..."):
            test_model(model)
    else:
        st.warning("No model found. Please train the model first.")

# Image upload for prediction
uploaded_file = st.sidebar.file_uploader("Upload your image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    img_array = preprocess_image("temp_image.jpg")

    if model:
        try:
            prediction = model.predict(img_array)
            result = 'Cancerous' if prediction[0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result:")
            st.write(f"The model predicts the image is: **{result}**")

            heatmap = generate_gradcam(model, img_array)
            superimposed_img = display_gradcam(img_array[0], heatmap)

            st.image("temp_image.jpg", caption='Uploaded Image', use_container_width=True)
            st.image(superimposed_img, caption='Superimposed Grad-CAM', use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    os.remove("temp_image.jpg")

# Mobile Upload Option
st.sidebar.header("Take a Picture")
photo = st.sidebar.file_uploader("Capture a photo", type=["jpg", "jpeg", "png"])

if photo is not None:
    with open("captured_image.jpg", "wb") as f:
        f.write(photo.getbuffer())

    img_array = preprocess_image("captured_image.jpg")

    if model:
        try:
            prediction = model.predict(img_array)
            result = 'Cancerous' if prediction[0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result for Captured Image:")
            st.write(f"The model predicts the image is: **{result}**")

            heatmap = generate_gradcam(model, img_array)
            superimposed_img = display_gradcam(img_array[0], heatmap)

            st.image("captured_image.jpg", caption='Captured Image', use_container_width=True)
            st.image(superimposed_img, caption='Superimposed Grad-CAM for Captured Image', use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    os.remove("captured_image.jpg")

# Clear cache button
if st.button("Clear Cache"):
    st.cache_data.clear()  # Clear the cache
    st.success("Cache cleared successfully!")
