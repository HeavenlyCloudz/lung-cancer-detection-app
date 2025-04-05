import streamlit as st
import snowflake.connector
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
MODEL_FILE = 'lung_cancer_detection_model.keras'
BATCH_SIZE = 32
base_data_dir = os.path.join(os.getcwd(), 'data')
train_data_dir = os.path.join(base_data_dir, 'train')
val_data_dir = os.path.join(base_data_dir, 'val')
test_data_dir = os.path.join(base_data_dir, 'test')
last_conv_layer_name = 'top_conv'

# Function to establish Snowflake connection
def get_snowflake_connection():
    return snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )

# Function to save prediction to Snowflake
def save_prediction_to_snowflake(image_path, prediction, confidence):
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO predictions (image_path, prediction, confidence) VALUES (%s, %s, %s)",
            (image_path, prediction, confidence)
        )
        conn.commit()
        st.success("Prediction saved to Snowflake.")
    except Exception as e:
        st.error(f"Error saving prediction: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# Function to calculate class weights
def calculate_class_weights(train_generator):
    labels = train_generator.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    return class_weights_dict

# Focal Loss Function
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

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

def preprocess_image(img_path):
    try:
        img = Image.open(img_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize image to (224, 224)
        img = img.resize((224, 224))
        img_array = np.asarray(img, dtype=np.float32)

        # Normalize the image data using EfficientNet's preprocess_input
        img_array = preprocess_input(img_array)

        # Expand dimensions to fit the model's input shape (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"Error processing image: {str(e)}")  # More detailed error message for debugging
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

# Training function
def train_model(model):
    train_generator, val_generator = load_data(train_data_dir, val_data_dir, BATCH_SIZE)
    if not train_generator or not val_generator:
        return
    
    class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    imbalance_ratio = max(class_weights) / sum(class_weights)
    
    loss_function = focal_loss(alpha=0.25, gamma=2.0) if imbalance_ratio > 1.5 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, class_weight=class_weights)
    
    model.save(MODEL_FILE)
    st.success("Model trained and saved successfully!")
    plot_training_history(history)


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
    try:
        # Creating the Grad-CAM model
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)

            # If pred_index is not provided, use the class with the highest probability
            if pred_index is None:
                pred_index = tf.argmax(preds[0])  # Index of the predicted class

            class_channel = preds[:, pred_index]  # Class channel to compute gradient for

        # Compute the gradients
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # Pool gradients over spatial dimensions

        # Get the output from the last convolutional layer
        last_conv_layer_output = last_conv_layer_output[0]

        # Compute the Grad-CAM heatmap
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0)  # ReLU activation on the heatmap

        # Normalize the heatmap to the range [0, 1]
        heatmap /= tf.reduce_max(heatmap) if tf.reduce_max(heatmap) > 0 else 1

        # Return the resized heatmap to match the input image dimensions
        return cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))

    except Exception as e:
        st.error(f"Error generating Grad-CAM heatmap: {str(e)}")
        return None

def display_gradcam(img, heatmap, alpha=0.4):
    try:
        # Convert the heatmap to 8-bit (0-255) format
        heatmap = np.uint8(255 * heatmap)

        # Use the new colormap API from Matplotlib (avoid deprecated `get_cmap`)
        jet = plt.cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]  # Get the RGB values
        jet_heatmap = jet_colors[heatmap]  # Apply the colormap to the heatmap
        jet_heatmap = np.uint8(jet_heatmap * 255)  # Convert to 8-bit format
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Resize the heatmap to match the input image dimensions
        jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))

        # Superimpose the heatmap on the original image
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
st.markdown("Visit my [GitHub](https://github.com/HeavenlyCloudz/lung-cancer-detection-app) repository for insight on my code.")

# Sidebar controls
st.sidebar.title("Controls🎮")

# Load the model
model = load_model_file()

# Hyperparameter inputs
epochs = st.sidebar.number_input("Number of epochs for training", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=BATCH_SIZE)

# Button to train model
if st.sidebar.button("Train Model"):
    with st.spinner("Training the model🤖..."):
        # Load training and validation data
        train_generator, val_generator = load_data(train_data_dir, val_data_dir, BATCH_SIZE)

        if train_generator is not None and val_generator is not None:
            # Compute class weights
            y_train = train_generator.classes
            class_labels = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
            class_weights = {i: weights[i] for i in range(len(class_labels))}

            # Check for class imbalance
            class_0_count = np.sum(y_train == 0)
            class_1_count = np.sum(y_train == 1)
            imbalance_ratio = max(class_0_count, class_1_count) / min(class_0_count, class_1_count) if min(class_0_count, class_1_count) > 0 else 1

            # Set loss function based on imbalance ratio
            if imbalance_ratio > 1.5:
                loss_function = focal_loss(alpha=0.25, gamma=2.0)
                st.sidebar.write(f"Detected significant class imbalance (ratio: {imbalance_ratio:.2f}). Using Focal Loss.")
            else:
                loss_function = 'binary_crossentropy'
                st.sidebar.write(f"Class balance is acceptable (ratio: {imbalance_ratio:.2f}). Using Binary Cross-Entropy.")

            # Add ReduceLROnPlateau Callback
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

            # Compile the model with the selected loss function
            model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

            # Train the model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=[reduce_lr]
            )

            # Save model
            model.save(MODEL_FILE)
            st.success("Model trained and saved successfully!")
            plot_training_history(history)
        else:
            st.error("Error loading training/validation data.")

# Button to test model
if st.sidebar.button("Test Model"):
    if model:
        with st.spinner("Testing the model📝..."):
            test_model(model)
    else:
        st.warning("No model found. Please train the model first.")

# Function to process and predict image
def process_and_predict(image_path, model):
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)

        if processed_image is not None and model:
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

            # Add description for cancerous result
            if result == 'Cancerous':
                st.write("**Note:** The model has determined this CT scan to stipulate the presence of cancer. Please consult with a health professional and other experts on these results.")

            if result == 'Non-Cancerous':
                st.write("**Note:** The model has determined this CT scan to be exempt from the presence of cancer. However, please continue to consult a health professional and other experts on these results.")

                # Symptoms checkboxes
                symptoms = [
                    "Persistent cough",
                    "Shortness of breath",
                    "Chest pain",
                    "Fatigue",
                    "Weight loss",
                    "Wheezing",
                    "Coughing up blood"
                ]
    
                # Multi-select for symptoms
                selected_symptoms = st.multiselect("Please select any symptoms you are experiencing:", symptoms)
    
                # Done button
                if st.button("Done"):
                    # Check how many symptoms are selected
                    if len(selected_symptoms) > 3:
                        st.warning("Even if it isn't cancer according to the model, these symptoms could point to other possible illnesses. Please contact medical support.")
                    elif len(selected_symptoms) > 0:
                        st.success("You have selected a manageable number of symptoms. Monitor your health and consult a healthcare provider if necessary.")
                    else:
                        st.info("No symptoms selected. If you are feeling unwell, please consult a healthcare provider.")
            
            # Generate Grad-CAM heatmap
            try:
                heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)

                if heatmap is not None:
                    uploaded_image = Image.open(image_path)  # Open with PIL

                    # Convert PIL image to numpy array for OpenCV compatibility
                    uploaded_image_np = np.array(uploaded_image)

                    superimposed_img = display_gradcam(uploaded_image_np, heatmap)

                    # Show images
                    st.image(image_path, caption='Uploaded Image', use_container_width=True)

                    if superimposed_img is not None:
                        st.image(superimposed_img, caption='Superimposed Grad-CAM', use_container_width=True)
                    else:
                        st.warning("Grad-CAM generation failed.")

                    uploaded_image.close()  # Close the PIL image
                else:
                    st.warning("Grad-CAM generation returned None.")

            except Exception as e:
                st.error(f"Error displaying Grad-CAM: {str(e)}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

    finally:
        # Ensure cleanup of the image file
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                st.warning(f"Error removing image file: {str(e)}")


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

# Initialize session state for layer visibility
if 'show_layers' not in st.session_state:
    st.session_state.show_layers = False

# Function to print layer names
def print_layer_names(model):
    try:
        layer_names = [layer.name for layer in model.layers]
        return layer_names
    except Exception as e:
        st.error(f"Error retrieving layer names: {str(e)}")
        return []

# Button to toggle visibility of layer names
if st.sidebar.button("Toggle Layer Names"):
    st.session_state.show_layers = not st.session_state.show_layers

# Displaying the layer names if the state is True
if st.session_state.show_layers:
    layer_names = print_layer_names(model)
    st.write("Layer names in the model:")
    st.text("\n".join(layer_names))
