import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load the model
model_file = os.path.join('streamlit_project', 'lung_cancer_detection_model.h5')
model = load_model(model_file)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Preprocess the image
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (150, 150))
    img_array = np.expand_dims(img, axis=0)
    return img_array / 255.0

# Generate the Grad-CAM
def generate_gradcam(model, img_array):
    last_conv_layer = model.layers[4]
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, last_conv_layer.output])

    with tf.GradientTape() as tape:
        model_output, last_conv_layer_output = grad_model(img_array)
        class_id = tf.argmax(model_output[0])
        grads = tape.gradient(model_output[:, class_id], last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (150, 150))
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to [0, 255]
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

# Streamlit UI
st.title("Lung Cancer Detection")
st.markdown(
    """
    <style>
    .section {
        background-image: url('https://jnj-content-lab2.brightspotcdn.com/dims4/default/78c6313/2147483647/strip/false/crop/1440x666+0+0/resize/1440x666!/quality/90/?url=https%3A%2F%2Fjnj-production-jnj.s3.us-east-1.amazonaws.com%2Fbrightspot%2F1b%2F32%2F2e138abbf1792e49103c9e3516a8%2Fno-one-would-believe-me-when-i-suspected-i-had-lung-cancer-0923-new.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        padding: 40px;
        min-height: 400px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
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

# Image upload for prediction
uploaded_file = st.sidebar.file_uploader("Upload your image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        img_array = preprocess_image("temp_image.jpg")

        # Ensure the input shape is correct
        if img_array.shape != (1, 150, 150, 3):
            st.error("Input shape is incorrect for the model.")
        else:
            prediction = model.predict(img_array)
            result = 'Cancerous' if prediction[0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result:")
            st.write(f"The model predicts the image is: **{result}**")

            heatmap = generate_gradcam(model, img_array)
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map to heatmap
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Overlay heatmap on the original image
            original_image = cv2.imread("temp_image.jpg")
            original_image = cv2.resize(original_image, (150, 150))
            superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_img, 0.4, 0)

            st.image(superimposed_img, caption='Overlayed Grad-CAM', use_container_width=True)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

    os.remove("temp_image.jpg")
