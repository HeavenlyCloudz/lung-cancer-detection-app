import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import class_weight
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

# Set the last convolutional layer name for Grad-CAM 
last_conv_layer_name = 'top_conv'

# Focal Loss Function
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_factor * tf.pow((1 - p_t), gamma)
        return focal_weight * bce
    return loss

# Compute Class Weights
def compute_class_weights(generator):
    labels = generator.classes
    class_labels, counts = np.unique(labels, return_counts=True)
    weights = class_weight.compute_class_weight('balanced', classes=class_labels, y=labels)
    imbalance_ratio = max(counts) / min(counts)
    return {i: weights[i] for i in range(len(class_labels))}, imbalance_ratio
    

def create_efficientnet_model(train_generator, input_shape=(224, 224, 3), num_classes=1):
    base_model = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)

    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False  

    # Unfreeze last 50 layers for fine-tuning
    for layer in base_model.layers[-50:]:
        layer.trainable = True  

    # Use base_model.input directly
    x = base_model.output

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)  

    # Output layer
    predictions = layers.Dense(1, activation='sigmoid')(x)  

    model = Model(inputs=base_model.input, outputs=predictions)

    # Calculate class weights
    class_weights = calculate_class_weights(train_generator)

    # Calculate imbalance ratio
    total_class_weight = sum(class_weights.values())
    imbalance_ratio = max(class_weights.values()) / total_class_weight

    # Decide whether to use focal loss or binary crossentropy based on imbalance ratio
    if imbalance_ratio > 1.5:  # Adjust the threshold based on your data
        loss_function = focal_loss(alpha=0.25, gamma=2.0)
    else:
        loss_function = 'binary_crossentropy'

    # Use SGD with momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)

    # Compile model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return model

model = create_efficientnet_model(train_generator)
model.summary()

def preprocess_image(img_path):
    try:
        img = Image.open(img_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize image to (224, 224)
        img = img.resize((IMAGE_HEIGHT, IMAGE_WIDTH))

        # Convert to numpy array
        img_array = np.asarray(img, dtype=np.float32)

        # Preprocess the image using EfficientNet's preprocess_input
        img_array = preprocess_input(img_array)

        # Expand dimensions to fit the model's input shape
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

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

# Load model from file
def load_model_file():
    if os.path.exists(MODEL_FILE):
        try:
            model = load_model(MODEL_FILE, custom_objects={"focal_loss": focal_loss})  # Load with focal loss
            
            # Use the same optimizer as in `create_efficientnet_model`
            optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)

            model.compile(optimizer=optimizer, 
                          loss=focal_loss(alpha=0.25, gamma=2.0), 
                          metrics=['accuracy'])
            
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("No saved model found.")
        return None


def print_layer_names():
    try:
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        layer_names = [layer.name for layer in base_model.layers]
        return layer_names
    except Exception as e:
        st.error(f"Error in print_layer_names: {str(e)}")
        return []

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
        tp = cm[1, 1]  
        fp = cm[0, 1]  
        fn = cm[1, 0]  

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
    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) if tf.reduce_max(heatmap) > 0 else 1

    return cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))

def display_gradcam(img, heatmap, alpha=0.4):
    try:
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = np.uint8(jet_heatmap * 255)
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2BGR)
        jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
        superimposed_img = cv2.addWeighted(jet_heatmap, alpha, img, 1 - alpha, 0)
        return superimposed_img
    except Exception as e:
        st.error(f"Error displaying Grad-CAM: {str(e)}")
        return None

# Streamlit UI
st.title("Lung Cancer Detection💻")
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

# Show the model summary
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))  # Capture the summary
st.text('\n'.join(model_summary))  # Display the summary in Streamlit

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
        model = create_efficientnet_model()  # Create a new EfficientNetB0 model
        train_generator, val_generator = load_data(train_data_dir, val_data_dir, batch_size)

        # Compute class weights
        if train_generator is not None and val_generator is not None:
            y_train = train_generator.classes
            class_labels = np.unique(y_train)
            weights = class_weight.compute_class_weight('balanced', classes=class_labels, y=y_train)
            class_weights = {i: weights[i] for i in range(len(class_labels))}

            # Check if class imbalance exists by comparing the class frequencies
            class_0_count = np.sum(y_train == 0)
            class_1_count = np.sum(y_train == 1)
            imbalance_ratio = max(class_0_count, class_1_count) / min(class_0_count, class_1_count) if min(class_0_count, class_1_count) > 0 else 1

            # If imbalance ratio is high, use Focal Loss, otherwise use Binary Cross-Entropy
            if imbalance_ratio > 1.5:
                loss_function = focal_loss(alpha=0.25, gamma=2.0)  # Use your Focal Loss
                st.sidebar.write(f"Detected significant class imbalance (ratio: {imbalance_ratio:.2f}). Using Focal Loss.")
            else:
                loss_function = 'binary_crossentropy'  # Use Binary Cross-Entropy
                st.sidebar.write(f"Class balance is acceptable (ratio: {imbalance_ratio:.2f}). Using Binary Cross-Entropy.")

            # **Add ReduceLROnPlateau Callback**
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

            # Compile the model with the selected loss function
            model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

            # Train the model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=[reduce_lr]  # **Include ReduceLROnPlateau here**
            )

            # Save model
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

# Function to process and predict image
def process_and_predict(image_path, model, last_conv_layer_name):
    processed_image = preprocess_image(image_path)

    if processed_image is not None and model:
        try:
            # Make prediction
            prediction = model.predict(processed_image)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction  # Confidence Score
            confidence_percentage = confidence * 100  # Convert to percentage
            
            # Determine result label
            result = 'Cancerous' if prediction > 0.5 else 'Non-Cancerous'
            
            # Display Prediction Result
            st.subheader("Prediction Result:")
            st.write(f"**{result}**")
            st.write(f"**Confidence: {confidence_percentage:.2f}%**")  # Show confidence

            # Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)

            if heatmap is not None:
                uploaded_image = Image.open(image_path)  # Open with PIL
                superimposed_img = display_gradcam(uploaded_image, heatmap)

                # Show images
                st.image(image_path, caption='Uploaded Image', use_container_width=True)
                st.image(superimposed_img, caption='Superimposed Grad-CAM', use_container_width=True)

            # Close and delete the image file
            uploaded_image.close()
            os.remove(image_path)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            os.remove(image_path)  # Ensure cleanup even if there's an error


# Load Model
last_conv_layer_name = 'top_conv'

# Normal Image Upload

uploaded_file = st.sidebar.file_uploader("Upload your image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    temp_filename = f"temp_image.{file_extension}"

    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_and_predict(temp_filename, model, last_conv_layer_name)

# Mobile Capture Option
st.sidebar.header("Take a Picture")
photo = st.sidebar.file_uploader("Capture a photo", type=["jpg", "jpeg", "png"])
if photo is not None:
    file_extension = photo.name.split('.')[-1]
    captured_filename = f"captured_image.{file_extension}"

    with open(captured_filename, "wb") as f:
        f.write(photo.getbuffer())

    process_and_predict(captured_filename, model, last_conv_layer_name)

# Clear cache button
if st.button("Clear Cache"):
    st.cache_data.clear()  # Clear the cache
    st.success("Cache cleared successfully!🎯")

if st.sidebar.button("Show Layer Names"):
    st.write("Layer names in EfficientNetB0:")
    layer_names = print_layer_names()
    st.text("\n".join(layer_names))
