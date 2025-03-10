import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load the model if it exists
model_exists = os.path.exists('lung_cancer_detection_model.h5')
if model_exists:
    model = load_model('lung_cancer_detection_model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Recompile if necessary
    model.summary()  # Print model summary for debugging

# Preprocess the image
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read the image in original format
    if img.shape[-1] == 4:  # Check if the image has 4 channels (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
    elif len(img.shape) == 2:  # Check if the image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB by repeating the channel
    img = cv2.resize(img, (150, 150))  # Resize to the input shape expected by the model
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    return img_array / 255.0  # Normalize the image

# Generate the Grad-CAM
def generate_gradcam(model, img_array):
    last_conv_layer = model.layers[4]  # Access the last Conv2D layer
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, last_conv_layer.output])

    with tf.GradientTape() as tape:
        model_output, last_conv_layer_output = grad_model(img_array)
        class_id = tf.argmax(model_output[0])
        grads = tape.gradient(model_output[:, class_id], last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)  # Normalize
    heatmap = cv2.resize(heatmap.numpy(), (150, 150))  # Keep original size
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
    plt.savefig('training_history.png')  # Save the plot as an image
    plt.close()  # Close the plot to free memory

# Create and compile the CNN model
def create_cnn_model(input_shape=(150, 150, 3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    return model

# Function to train the model
def train_model():
    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load data
    train_generator = train_datagen.flow_from_directory('data/train', target_size=(150, 150),
                                                        batch_size=32, class_mode='binary')
    val_generator = val_datagen.flow_from_directory('data/val', target_size=(150, 150),
                                                    batch_size=32, class_mode='binary')

    # Create and compile the model
    model = create_cnn_model((150, 150, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Call the model with a dummy input to initialize it
    dummy_input = np.random.rand(1, 150, 150, 3)  # Create a dummy input
    model(dummy_input)  # Call the model with dummy input

    # Train the model
    history = model.fit(train_generator, steps_per_epoch=train_generator.samples // 32,
                        validation_data=val_generator, validation_steps=val_generator.samples // 32,
                        epochs=10)

    # Save the model
    model.save('lung_cancer_detection_model.h5')

    # Plot and save the training history
    plot_training_history(history)

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

# Headers of the website
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("Thank you for using ONCOSCAN")
st.write("CNNs are the preferred network for detecting lung cancer due to their ability to process image data. They can perform tasks such as classification, segmentation, and object recognition. In the case of lung cancer detection, CNNs have surpassed radiologists.")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Controls")

# Train model button
if st.sidebar.button("Train Model"):
    with st.spinner("Training the model..."):
        train_model()
    st.success("Model training complete!")

# Display training history if it exists
if os.path.exists('training_history.png'):
    st.subheader("Training History")
    st.image('training_history.png', caption='Training History', use_container_width=True)

# Image upload for prediction
uploaded_file = st.sidebar.file_uploader("Upload your image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    img_array = preprocess_image("temp_image.jpg")

    if model_exists:
        try:
            prediction = model.predict(img_array)
            result = 'Cancerous' if prediction[0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result:")
            st.write(f"The model predicts the image is: **{result}**")

            heatmap = generate_gradcam(model, img_array)
            st.image("temp_image.jpg", caption='Uploaded Image', use_container_width=True)
            st.image(heatmap, caption='Grad-CAM', use_container_width=True)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Model not found. Please train the model first.")

    os.remove("temp_image.jpg")

# Mobile Upload Option
st.sidebar.header("Take a Picture")
photo = st.sidebar.file_uploader("Capture a photo", type=["jpg", "jpeg", "png"])

if photo is not None:
    # Save the captured image temporarily
    with open("captured_image.jpg", "wb") as f:
        f.write(photo.getbuffer())

    # Preprocess the captured image
    img_array = preprocess_image("captured_image.jpg")

    if model_exists:
        try:
            prediction = model.predict(img_array)
            result = 'Cancerous' if prediction[0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result for Captured Image:")
            st.write(f"The model predicts the image is: **{result}**")

            heatmap = generate_gradcam(model, img_array)
            st.image("captured_image.jpg", caption='Captured Image', use_container_width=True)
            st.image(heatmap, caption='Grad-CAM', use_container_width=True)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Model not found. Please train the model first.")

    os.remove("captured_image.jpg")