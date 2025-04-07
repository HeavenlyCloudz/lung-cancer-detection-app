import streamlit as st
import snowflake.connector
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.utils import Sequence
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.cm as cm

# Custom Data Generator Class
class CustomDataGenerator(Sequence):
    def __init__(self, directory, batch_size, input_shape):
        self.directory = directory
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.image_paths = []
        self.labels = []
        self.class_map = {
            'normal': 0,
            'benign': 1,
            'adenocarcinoma': 2,
            'squamous_cell_carcinoma': 3,
            'large_cell_carcinoma': 4,
            'malignant': 5
        }
        self._load_data()

    def _load_data(self):
        for tumor_status in os.listdir(self.directory):
            status_dir = os.path.join(self.directory, tumor_status)
            if os.path.isdir(status_dir):
                for class_name in os.listdir(status_dir):
                    class_dir = os.path.join(status_dir, class_name)
                    if os.path.isdir(class_dir):
                        for img_file in os.listdir(class_dir):
                            img_path = os.path.join(class_dir, img_file)
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_map[class_name])

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_x = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([img_to_array(load_img(img_path, target_size=self.input_shape)) for img_path in batch_x])
        
        y_cancerous = np.array([(1 if label > 1 else 0) for label in self.labels])
        y_cancer_type = np.array(self.labels)
        y_non_cancerous = np.array([(1 if label == 0 else 0) for label in self.labels])
        
        return X, [y_cancerous, y_cancer_type, y_non_cancerous]

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

is_new_model = False  # Global flag to indicate if a new model is created

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

def get_snowflake_connection():
    required_env_vars = [
        'SNOWFLAKE_USER', 
        'SNOWFLAKE_PASSWORD', 
        'SNOWFLAKE_ACCOUNT', 
        'SNOWFLAKE_WAREHOUSE', 
        'SNOWFLAKE_DATABASE', 
        'SNOWFLAKE_SCHEMA'
    ]
    for var in required_env_vars:
        if os.getenv(var) is None:
            raise EnvironmentError(f"Environment variable {var} is not set.")
    return snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )

# Function to calculate class weights
def calculate_class_weights(train_generator):
    # Assuming that the class labels are integers
    labels = train_generator.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    return class_weights_dict

# Function to compute imbalance ratio
def compute_class_weights(train_generator):
    class_weights = calculate_class_weights(train_generator)
    total_class_weight = sum(class_weights.values())
    imbalance_ratio = max(class_weights.values()) / total_class_weight
    return class_weights, imbalance_ratio

# Focal Loss Function
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def create_efficientnet_model(input_shape=(224, 224, 3), learning_rate=1e-3):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)

    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last 50 layers for fine-tuning
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    # Build the model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    # First output: Binary classification for cancerous vs non-cancerous
    cancerous_output = layers.Dense(1, activation='sigmoid', name='cancerous_output')(x)
    
    # Second output: Predict cancer type if the image is cancerous
    cancer_type_output = layers.Dense(4, activation='softmax', name='cancer_type_output')(x)  # 4 types of cancer
    
    # Third output: Predict non-cancerous type (benign or normal)
    non_cancerous_type_output = layers.Dense(2, activation='softmax', name='non_cancerous_type_output')(x)  # 2 types: benign or normal
    
    model = models.Model(inputs=base_model.input, outputs=[cancerous_output, cancer_type_output, non_cancerous_type_output])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'cancerous_output': 'binary_crossentropy',  # For binary output
            'cancer_type_output': 'categorical_crossentropy',  # For cancerous types
            'non_cancerous_type_output': 'categorical_crossentropy'  # For non-cancerous types
        },
        metrics={
            'cancerous_output': 'accuracy',
            'cancer_type_output': 'accuracy',
            'non_cancerous_type_output': 'accuracy'
        }
    )

    return model
    
# Function to load model from file
def load_model_file():
    global is_new_model
    if os.path.exists(MODEL_FILE):
        try:
            model = tf.keras.models.load_model(MODEL_FILE, custom_objects={"focal_loss": focal_loss})
            for layer in model.layers[-50:]:
                layer.trainable = True
            st.success("✅ Model loaded")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("No saved model found. Creating a new model.")
        is_new_model = True
        return create_efficientnet_model()  # Ensure you have defined this function

# Load or create the model
model = load_model_file()

if model is None:
    st.error("Failed to load model. Please check the model file.")
    st.stop()  # Stops further execution of the Streamlit app

def preprocess_image(img_path):
    try:
        img = Image.open(img_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize image to (224, 224)
        img = img.resize((224, 224))
        print(f"Resized image size: {img.size}")  # Debugging print

        # Convert image to numpy array
        img_array = np.asarray(img, dtype=np.float32)

        # Ensure the correct shape
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Unexpected shape after resizing: {img_array.shape}")

        # Normalize the image data using EfficientNet's preprocess_input
        img_array = preprocess_input(img_array)

        # Expand dimensions to fit the model's input shape (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Image array shape after expanding: {img_array.shape}")  # Debugging print

        return img_array

    except Exception as e:
        print(f"Error processing image: {str(e)}")  # More detailed error message for debugging
        return None

# Define the predict function
def predict(input_tensor):
    if model is not None:
        return model(input_tensor)
    else:
        raise ValueError("Model not loaded. Cannot make predictions.")

# Example usage of the prediction function
def make_prediction(image_path):
    try:
        input_tensor = preprocess_image(image_path)  # Preprocess the image
        predictions = predict(input_tensor)  # Call the predict function
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Main function to run predictions
def main():
    # Example usage of the predict function
    image_tensor = tf.random.uniform((1, 224, 224, 3))  # Simulate an image tensor
    result = predict(image_tensor)
    print(result)


# Load training and validation data using CustomDataGenerator
def load_data(train_dir, val_dir, batch_size):
    try:
        train_generator = CustomDataGenerator(train_dir, batch_size, (IMAGE_HEIGHT, IMAGE_WIDTH))
        val_generator = CustomDataGenerator(val_dir, batch_size, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        return train_generator, val_generator
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def print_layer_names():
    try:
        base_model = EfficientNetB0(include_top=False, weights='', input_shape=(224, 224, 3))
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

# Training function
def train(train_dir, val_dir):
    global model  # Use the global model variable

    train_generator, val_generator = load_data(train_dir, val_dir, BATCH_SIZE)

    if not train_generator or not val_generator:
        st.error("Failed to load training or validation data.")
        return

    # Prepare labels
    try:
        # Get the class indices from the generator
        y_train = train_generator.classes
        y_val = val_generator.classes
        
        # Prepare binary labels for cancerous output (0 for non-cancerous, 1 for cancerous)
        y_cancerous = (y_train >= 1).astype(int)  # Assuming class 0 is non-cancerous, 1+ are cancerous
        
        # Prepare cancer type labels
        # Classes:
        # 1 = Adenocarcinoma
        # 2 = Squamous Cell Carcinoma
        # 3 = Large Cell Carcinoma
        # 4 = Malignant (general)
        # 0 = Benign
        # 5 = Normal
        y_cancer_type = np.where(y_cancerous == 1, y_train, 0)  # Non-cancerous to 0
        y_cancer_type[y_cancerous == 0] = 5  # Assign non-cancerous to class 5 (Normal)

        # Ensure malignant cases are represented correctly, you might need to adjust y_train accordingly
        y_cancer_type[y_cancer_type == 1] = 4  # If class 1 is malignant, assign it to class 4

        # One-hot encode the cancer type labels (0-5: 0=Benign, 1=Adenocarcinoma, 2=Squamous, 3=Large Cell, 4=Malignant, 5=Normal)
        encoder_cancer_type = OneHotEncoder(sparse=False)
        y_cancer_type_encoded = encoder_cancer_type.fit_transform(y_cancer_type.reshape(-1, 1))

        # Combine labels into a tuple for model training
        y_labels = (y_cancerous, y_cancer_type_encoded)

    except Exception as e:
        st.error(f"Error preparing labels: {str(e)}")
        return  # Exit the function

    class_weights = calculate_class_weights(train_generator)
    imbalance_ratio = max(class_weights.values()) / sum(class_weights.values())
    loss_function = focal_loss(alpha=0.25, gamma=2.0) if imbalance_ratio > 1.5 else 'categorical_crossentropy'

    # Compile the model if it hasn't been compiled yet
    if not model._is_compiled:  # Check if the model is compiled
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss={
            'cancerous_output': 'binary_crossentropy',  # For binary classification of cancer presence
            'non_cancerous_output': 'categorical_crossentropy'  # For types of cancer or non-cancerous
        }, metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weights
    )

    model.save(MODEL_FILE)
    st.success("Model saved successfully!")
    st.write("Training completed.")
    

def test_model(model):
    try:
        # Use CustomDataGenerator for the test data
        test_generator = CustomDataGenerator(test_data_dir, BATCH_SIZE, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Get true labels
        true_labels = test_generator.labels  # Assuming this returns the correct labels

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(test_generator)
        st.sidebar.write(f"Test Loss: {test_loss:.4f}")
        st.sidebar.write(f"Test Accuracy: {test_accuracy:.4f}")

        # Get predictions
        y_pred = model.predict(test_generator)

        # Check if predictions are None or empty
        if y_pred is None or len(y_pred) == 0:
            st.error("Prediction failed. Model output is None or empty.")
            return

        # Extract binary and categorical predictions
        y_pred_binary = (y_pred[0] > 0.5).astype(int)  # Binary prediction for cancer presence
        y_pred_categorical = np.argmax(y_pred[1], axis=1)

        # Combine predictions for confusion matrix
        combined_preds = np.where(y_pred_binary.flatten() == 0, 5, y_pred_categorical + 1)

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, combined_preds)

        # Calculate precision and recall for each class
        precision = {}
        recall = {}
        f1_score = {}
        for i in range(len(np.unique(true_labels))):
            tp = cm[i, i]  # True Positives
            fp = cm[:, i].sum() - tp  # False Positives
            fn = cm[i, :].sum() - tp  # False Negatives
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        # Display average metrics
        average_precision = np.mean(list(precision.values()))
        average_recall = np.mean(list(recall.values()))
        average_f1 = np.mean(list(f1_score.values()))

        st.sidebar.write(f"Average Precision: {average_precision:.4f}")
        st.sidebar.write(f"Average Recall: {average_recall:.4f}")
        st.sidebar.write(f"Average F1 Score: {average_f1:.4f}")

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=['Benign', 'Adenocarcinoma', 'Squamous Cell Carcinoma', 
                                  'Large Cell Carcinoma', 'Malignant', 'Normal'], 
                     yticklabels=['Benign', 'Adenocarcinoma', 'Squamous Cell Carcinoma', 
                                  'Large Cell Carcinoma', 'Malignant', 'Normal'], 
                     ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during testing: {str(e)}")
        
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            
            if pred_index is None:
                # Get the index of the predicted class if none is provided
                pred_index = tf.argmax(preds[0])

            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        last_conv_layer_output = last_conv_layer_output[0]  # Remove batch dimension
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) if tf.reduce_max(heatmap) > 0 else 1  # Normalize heatmap

        # Resize the heatmap to match the original image size (if necessary)
        heatmap = cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))
        return heatmap

    except Exception as e:
        st.error(f"Error generating Grad-CAM heatmap: {str(e)}")
        return None

def display_gradcam(img_array, heatmap, alpha=0.4):
    try:
        # Ensure img_array is RGB and convert to float32
        if img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError("Input image must have 3 channels (RGB).")
        
        img_array = img_array.astype(np.float32) / 255.0  # Normalize image to [0, 1]

        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))

        # Normalize heatmap to be in the range [0, 255]
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        jet = plt.cm.jet(np.arange(256))[:, :3]  # Get jet colormap
        jet_heatmap = jet[heatmap_normalized]  # Apply colormap to heatmap
        jet_heatmap = np.uint8(jet_heatmap * 255)  # Convert to uint8

        # Ensure both images are uint8 before blending
        img_array = np.uint8(img_array * 255)  # Convert back to uint8
        superimposed_img = cv2.addWeighted(img_array, 1 - alpha, jet_heatmap, alpha, 0)

        return superimposed_img  # Return the blended image

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

# Add input for number of evaluations during testing
eval_epochs = st.sidebar.number_input("Number of evaluations for testing", min_value=1, max_value=10, value=1)

# Button to train model
if st.sidebar.button("Train Model"):
    if model is not None:
        with st.spinner("Training the model🤖..."):
            train_generator, val_generator = load_data(train_data_dir, val_data_dir, BATCH_SIZE)

            # Ensure generators are loaded
            if train_generator is not None and val_generator is not None:
                y_train = train_generator.classes
                class_labels = np.unique(y_train)
                weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
                class_weights = {i: weights[i] for i in range(len(class_labels))}

                # Calculate imbalance ratio for 6 classes
                class_counts = np.bincount(y_train)
                imbalance_ratio = max(class_counts) / min(class_counts[class_counts > 0])

                # Determine loss function based on imbalance
                if imbalance_ratio > 1.5:
                    loss_function = focal_loss(alpha=0.25, gamma=2.0)
                    st.sidebar.write(f"Detected significant class imbalance (ratio: {imbalance_ratio:.2f}). Using Focal Loss.")
                else:
                    loss_function = 'categorical_crossentropy'
                    st.sidebar.write(f"Class balance is acceptable (ratio: {imbalance_ratio:.2f}). Using Categorical Cross-Entropy.")

                # Add ReduceLROnPlateau Callback
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

                # Compile the model with the selected loss function
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                model.compile(optimizer=optimizer, loss={
                    'cancerous_output': 'binary_crossentropy',
                    'non_cancerous_output': 'categorical_crossentropy'
                }, metrics=['accuracy'])

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

                # Add a download button for the model file after training completes
                with open(MODEL_FILE, "rb") as f:
                    model_data = f.read()
                st.download_button(
                    label="Download Trained Model",
                    data=model_data,
                    file_name=MODEL_FILE,
                    mime="application/octet-stream"
                )
    else:
        st.error("Model is not available for training. Please check model initialization.")

# Button to test model
if st.sidebar.button("Test Model"):
    if model:
        with st.spinner("Testing the model📝..."):
            for _ in range(eval_epochs):  # Repeat testing as per user input
                test_model(model)
    else:
        st.warning("No model found. Please train the model first.")

# Updated label mapping to reflect the new classification scheme
label_mapping = {
    0: 'Benign',
    1: 'Adenocarcinoma',
    2: 'Squamous Cell Carcinoma',
    3: 'Large Cell Carcinoma',
    4: 'Malignant',
    5: 'Normal'
}

def process_and_predict(image_path, model, label_mapping, last_conv_layer_name):
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)

        if processed_image is None:
            st.error("Failed to process the image. Please try again.")
            return

        if model:
            # Make prediction
            prediction = model.predict(processed_image)

            if prediction is None or len(prediction) == 0:
                st.error("Prediction failed. Please check the input image or try again.")
                return

            # Confidence and threshold logic
            confidence = np.max(prediction[0]) * 100  # Get the highest confidence score
            cancerous_threshold = 0.5  # Set the threshold for classification as cancerous

            # Determine category
            if confidence > cancerous_threshold:
                category = 'Cancerous'
                predicted_index = np.argmax(prediction[1])  # Get the cancer type index
                predicted_label = label_mapping[predicted_index] if label_mapping else str(predicted_index)
            else:
                category = 'Non-Cancerous'
                predicted_label = 'Normal'  # Default for non-cancerous

            # Display the result
            st.subheader("Prediction Result:")
            st.write(f"**Category:** {category}")
            st.write(f"**Type:** {predicted_label}")
            st.write(f"**Confidence: {confidence:.2f}%**")

            # Notes and symptoms
            if category == 'Cancerous':
                st.write("**Note:** The model has determined this CT scan to stipulate the presence of cancer. Please consult with a health professional.")
            else:
                st.write("**Note:** The model has determined this CT scan to be non-cancerous. However, please consult a health professional.")

                # Symptoms for non-cancerous cases
                symptoms = [
                    "Persistent cough", "Shortness of breath", "Chest pain",
                    "Fatigue", "Weight loss", "Wheezing", "Coughing up blood"
                ]

                selected_symptoms = st.multiselect("Please select any symptoms you are experiencing:", symptoms)

                if st.button("Done"):
                    if len(selected_symptoms) > 3:
                        st.warning("Even if it isn't cancer according to the model, these symptoms could point to other possible illnesses.")
                    elif len(selected_symptoms) == 3:
                        st.warning("These symptoms could possibly point to other diseases as well.")
                    elif len(selected_symptoms) > 0:
                        st.success("Monitor your health and consult a healthcare provider if necessary.")
                    else:
                        st.info("No symptoms selected. If you are feeling unwell, please consult a healthcare provider.")

            # Generate Grad-CAM heatmap
            try:
                heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)

                if heatmap is not None:
                    uploaded_image = Image.open(image_path).convert('RGB')
                    uploaded_image_np = np.array(uploaded_image)

                    superimposed_img = display_gradcam(uploaded_image_np, heatmap)

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

if st.sidebar.button("Show Layer Names"):
    st.write("Layer names in EfficientNetB0:")
    layer_names = print_layer_names()
    st.text("\n".join(layer_names))

# Function to collect feedback
def collect_feedback():
    st.title(":rainbow[Feedback] Form")
    
    # Add a text area for the feedback
    feedback = st.text_area("Please share your feedback to improve this app💕", "", height=150)
    
    # Add a submit button
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback🫶!")
            # Save feedback to a file, database, or send it via email, etc.
            save_feedback(feedback)
        else:
            st.error("Please enter some feedback before submitting😡.")

# Function to save feedback (can be customized to store feedback anywhere)
def save_feedback(feedback):
    # Example: Save to a text file (or database)
    with open("user_feedback.txt", "a") as f:
        f.write(f"Feedback: {feedback}\n{'-'*50}\n")
    st.info("Your feedback has been recorded.")

# Show the feedback form
collect_feedback()

if __name__ == "__main__":
    main()
