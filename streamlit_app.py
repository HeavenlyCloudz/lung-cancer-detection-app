import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
MODEL_FILE = 'lung_cancer_detection_model.keras'
BATCH_SIZE = 32
base_data_dir = os.path.join(os.getcwd(), 'data')
train_data_dir = os.path.join(base_data_dir, 'train')
val_data_dir = os.path.join(base_data_dir, 'val')
test_data_dir = os.path.join(base_data_dir, 'test')

def create_model(num_classes=1):
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    input_tensor = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))  # Ensure consistent input shape
    x = base_model(input_tensor)

    # Use Flatten
    x = layers.Flatten()(x)  # Convert the output to a 1D vector

    # Force the output shape to 36992 using Reshape
    x = layers.Reshape((36992,))(x)  # Reshape to the desired size

    # Set the Dense layer to a reasonable number of units
    x = layers.Dense(256, activation='relu')(x)  # Use a reasonable number of units

    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='sigmoid')(x)  # Output layer for binary classification

    model = tf.keras.models.Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
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

# Preprocess the image for prediction
def preprocess_image(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))  # Load and resize image
        image_array = img_to_array(img)                                     # Convert to array
        image_array = np.expand_dims(image_array, axis=0)                  # Add batch dimension
        image_array = preprocess_input(image_array)                         # Preprocess

        print(f"Processed image shape: {image_array.shape}")  # Debug output
        return image_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Load training and validation data
def load_data(train_dir, val_dir, batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=batch_size,
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=batch_size,
            class_mode='binary'
        )

        return train_generator, val_generator
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


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

        # Calculate precision and recall
        tp = cm[1, 1]  # True Positives
        fp = cm[0, 1]  # False Positives
        fn = cm[1, 0]  # False Negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        st.sidebar.write(f"Precision: {precision:.4f}")
        st.sidebar.write(f"Recall: {recall:.4f}")
        st.sidebar.write(f"F1 Score: {f1_score:.4f}")

        # Plot confusion matrix
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

    # Normalize the heatmap
    heatmap /= tf.reduce_max(heatmap) if tf.reduce_max(heatmap) > 0 else 1

    return cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))

# Display Grad-CAM heatmap
def display_gradcam(img, heatmap, alpha=0.4):
    try:
        # Convert heatmap to uint8
        heatmap = np.uint8(255 * heatmap)

        # Apply colormap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]  # Get RGB values
        jet_heatmap = jet_colors[heatmap]

        # Convert to an image
        jet_heatmap = np.uint8(jet_heatmap * 255)  # Scale to 0-255
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Resize heatmap to match the original image size
        jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))

        # Combine heatmap with original image
        superimposed_img = cv2.addWeighted(jet_heatmap, alpha, img, 1 - alpha, 0)
        return superimposed_img
    except Exception as e:
        st.error(f"Error displaying Grad-CAM: {str(e)}")
        return None

# Streamlit UI
st.title("Lung Cancer Detection🖥️")
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
st.header("Thank you for using ONCO AI🌐")
st.write("CNNs are the preferred network for detecting lung cancer due to their ability to process image data. They can perform tasks such as classification, segmentation, and object recognition. In the case of lung cancer detection, CNNs have surpassed radiologists.")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("Visit [ONCO AI](https://readymag.website/u4174625345/5256774/) for more information.")

# Sidebar controls
st.sidebar.title("Controls🎮")

# Load the model
model = load_model_file()

# Hyperparameter inputs
epochs = st.sidebar.number_input("Number of epochs for training", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=BATCH_SIZE)

# Add input for number of evaluations during testing
eval_epochs = st.sidebar.number_input("Number of evaluations for testing", min_value=1, max_value=10, value=1)

# Button to train model
if st.sidebar.button("Train Model"):
    with st.spinner("Training the model🤖..."):
        model = create_densenet_model()  # Create a new DenseNet model
        train_generator, val_generator = load_data(train_data_dir, val_data_dir, batch_size)

        # Calculate class weights
        if train_generator is not None and val_generator is not None:
            y_train = train_generator.classes
            class_labels = np.unique(y_train)
            weights = class_weight.compute_class_weight('balanced', classes=class_labels, y=y_train)
            class_weights = {i: weights[i] for i in range(len(class_labels))}

            # Early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = model.fit(train_generator, validation_data=val_generator, 
                                epochs=epochs, class_weight=class_weights, 
                                callbacks=[early_stopping])  # Use early stopping
            model.save(MODEL_FILE)
            st.success("Model trained and saved successfully!")
            plot_training_history(history)

# Button to test model
if st.sidebar.button("Test Model"):
    if model:
        with st.spinner("Testing the model📝..."):
            for _ in range(eval_epochs):  # Repeat testing as per user input
                test_model(model)
    else:
        st.warning("No model found. Please train the model first.")

# Image upload for prediction
uploaded_file = st.sidebar.file_uploader("Upload your image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the image for prediction
    processed_image = preprocess_image("temp_image.jpg")
    if processed_image is not None and model:  # Check if processed_image is valid before prediction
        try:
            # Make prediction
            prediction = model.predict(processed_image)
            result = 'Cancerous' if prediction[0][0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result:")
            st.write(f"The model predicts the image is: **{result}**")

            # Get the last convolutional layer name for Grad-CAM
            last_conv_layer_name = 'conv5_block32_concat'  # Change this if using a different layer

            # Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)
            if heatmap is not None:
                uploaded_image = cv2.imread("temp_image.jpg")
                superimposed_img = display_gradcam(uploaded_image, heatmap)
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

    # Load the image for prediction
    processed_image = preprocess_image("captured_image.jpg")
    if processed_image is not None and model:  # Check if processed_image is valid before prediction
        try:
            # Make prediction
            prediction = model.predict(processed_image)
            result = 'Cancerous' if prediction[0][0] > 0.5 else 'Non-Cancerous'
            st.subheader("Prediction Result for Captured Image:")
            st.write(f"The model predicts the image is: **{result}**")

            # Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)
            if heatmap is not None:
                captured_image = cv2.imread("captured_image.jpg")
                superimposed_img = display_gradcam(captured_image, heatmap)
                st.image("captured_image.jpg", caption='Captured Image', use_container_width=True)
                st.image(superimposed_img, caption='Superimposed Grad-CAM for Captured Image', use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    os.remove("captured_image.jpg")

# Clear cache button
if st.button("Clear Cache"):
    st.cache_data.clear()  # Clear the cache
    st.success("Cache cleared successfully!")
