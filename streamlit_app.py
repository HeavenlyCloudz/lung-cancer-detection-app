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
from sklearn.utils.class_weight import compute_class_weight
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
test_data_dir = os.path.join(base_data_dir, 'test')

# Set the last convolutional layer name for Grad-CAM 
last_conv_layer_name = 'top_conv'

is_new_model = False  # Global flag to indicate if a new model is created

# Function to calculate class weights
def calculate_class_weights(train_generator):
    # Assuming that the class labels are integers, e.g., 0 and 1 for binary classification
    labels = train_generator.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    return class_weights_dict

# Function to compute imbalance ratio
def compute_class_weights(train_generator):
    # Extract class labels from generator
    y_train = train_generator.classes
    class_labels = np.unique(y_train)
    
    # Compute class weights
    weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
    class_weights = {int(label): weight for label, weight in zip(class_labels, weights)}

    # Compute imbalance ratio
    total_weight = sum(class_weights.values())
    imbalance_ratio = max(class_weights.values()) / total_weight

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

@st.cache_resource
def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=1, learning_rate=1e-3):
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
    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model  # Return the model

# Load model from file or create a new one
def load_model_file():
    global is_new_model
    if os.path.exists(MODEL_FILE):
        try:
            model = load_model(MODEL_FILE, custom_objects={"focal_loss": focal_loss})
            # Set last 50 layers as trainable
            for layer in model.layers[-50:]:
                layer.trainable = True
            st.success("✅")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("No saved model found. Creating a new model.")
        is_new_model = True  # Set the flag to indicate a new model is created
        return create_efficientnet_model()

# Load or create the model
model = load_model_file()

# Define the predict function with tf.function
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
def predict(input_tensor):
    return model(input_tensor)

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

# Main function to run predictions
def main():
    # Example usage of the predict function
    image_tensor = tf.random.uniform((1, 224, 224, 3))  # Simulate an image tensor
    result = predict(image_tensor)
    print(result)


def load_test_data(test_dir, batch_size):
    test_datagen = ImageDataGenerator(rescale=1./255)

    try:
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False  # important for evaluation
        )

        return test_generator
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None


def print_layer_names():
    try:
        base_model = EfficientNetB0(include_top=False, weights='', input_shape=(224, 224, 3))
        layer_names = [layer.name for layer in base_model.layers]
        return layer_names
    except Exception as e:
        st.error(f"Error in print_layer_names: {str(e)}")
        return []

# def plot_training_history(history):
    #try:
       # fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        #ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        #ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        #ax[0].set_title('Model Accuracy')
        #ax[0].set_xlabel('Epoch')
        #ax[0].set_ylabel('Accuracy')
        #ax[0].legend()

        #ax[1].plot(history.history['loss'], label='Train Loss')
        #ax[1].plot(history.history['val_loss'], label='Validation Loss')
        #ax[1].set_title('Model Loss')
        #ax[1].set_xlabel('Epoch')
        #ax[1].set_ylabel('Loss')
        #ax[1].legend()

        #plt.tight_layout()
        #st.pyplot(fig)
    #except Exception as e:
      #  st.error(f"Error plotting training history: {str(e)}")

# Training function
#def train(train_dir, val_dir):
    #global model  # Use the global model variable

    #train_generator, val_generator = load_data(train_dir, val_dir, BATCH_SIZE)

    #if not train_generator or not val_generator:
     #   st.error("Failed to load training or validation data.")
      #  return

    #class_weights = calculate_class_weights(train_generator)
    #imbalance_ratio = max(class_weights.values()) / sum(class_weights.values())
    #loss_function = focal_loss(alpha=0.25, gamma=2.0) if imbalance_ratio > 1.5 else 'binary_crossentropy'

    # Compile the model if it hasn't been compiled yet
    #if not model._is_compiled:  # Check if the model is compiled
     #   optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
      #  model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # Train the model
   # history = model.fit(
    #    train_generator,
     #   epochs=EPOCHS,
      #  validation_data=val_generator,
       # class_weight=class_weights
    #)

    #model.save(MODEL_FILE)
   # st.success("Model saved successfully!")
   # st.write("Training completed.")
    

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

        # Resize the heatmap to match the original image size
        heatmap = cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))
        return heatmap

    except Exception as e:
        st.error(f"Error generating Grad-CAM heatmap: {str(e)}")
        return None

def display_gradcam(img, heatmap, alpha=0.4):
    try:
        # Ensure img is in the correct format (BGR to RGB if needed)
        if img.shape[2] == 3:  # Check if the image has 3 channels
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        heatmap = np.uint8(255 * heatmap)

        # Use the updated method to get the colormap
        jet = plt.colormaps['jet']
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = np.uint8(jet_heatmap * 255)
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2BGR)

        # Resize the heatmap to the original image size
        jet_heatmap = cv2.resize(jet_heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

        superimposed_img = cv2.addWeighted(jet_heatmap, alpha, img_rgb, 1 - alpha, 0)
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
epochs = st.sidebar.number_input("Number of epochs for training🔋", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input("Batch size🎚️", min_value=1, max_value=64, value=BATCH_SIZE)

# Add input for number of evaluations during testing
eval_epochs = st.sidebar.number_input("Number of evaluations for testing🧾", min_value=1, max_value=10, value=1)

# Button to train model
# if st.sidebar.button("Train Model"):
    # if model is not None:
       # with st.spinner("Training the model🤖..."):
            #train_generator, val_generator = load_data(train_data_dir, val_data_dir, BATCH_SIZE)

            # Ensure generators are loaded
           # if train_generator is not None and val_generator is not None:
              #  y_train = train_generator.classes
               # class_labels = np.unique(y_train)
                #weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
               # class_weights = {i: weights[i] for i in range(len(class_labels))}

                # Calculate imbalance ratio
                #class_0_count = np.sum(y_train == 0)
                #class_1_count = np.sum(y_train == 1)
                #imbalance_ratio = (
                 #   max(class_0_count, class_1_count) / min(class_0_count, class_1_count)
                  #  if min(class_0_count, class_1_count) > 0
                   # else 1
                #)

                # Determine loss function based on imbalance
                #if imbalance_ratio > 1.5:
                 #   loss_function = focal_loss(alpha=0.25, gamma=2.0)
                  #  st.sidebar.write(f"Detected significant class imbalance (ratio: {imbalance_ratio:.2f}). Using Focal Loss.")
                #else:
                 #   loss_function = 'binary_crossentropy'
                  #  st.sidebar.write(f"Class balance is acceptable (ratio: {imbalance_ratio:.2f}). Using Binary Cross-Entropy.")

                # Add ReduceLROnPlateau Callback
                #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

                # Compile the model with the selected loss function
                #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                #model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

                # Train the model
                #history = model.fit(
                 #   train_generator,
                  #  validation_data=val_generator,
                   # epochs=epochs,
                    #class_weight=class_weights,
                    #callbacks=[reduce_lr]
                #)

                # Save model
               # model.save(MODEL_FILE)
               # st.success("Model trained and saved successfully!")
                #plot_training_history(history)

                # Add a download button for the model file after training completes
                #with open(MODEL_FILE, "rb") as f:
                 #   model_data = f.read()
               # st.download_button(
                #    label="Download Trained Model",
                 #   data=model_data,
                  #  file_name=MODEL_FILE,
                   # mime="application/octet-stream"
       
   # else:
       # st.error("Model is not available for training. Please check model initialization.")

# Button to test model
if st.sidebar.button("Test Model🏫"):
    if model:
        with st.spinner("Testing the model📝..."):
            for _ in range(eval_epochs):  # Repeat testing as per user input
                test_model(model)
    else:
        st.warning("No model found. Please train the model first.")

# Function to process and predict image
def process_and_predict(image_path, model, last_conv_layer_name):
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)

        if processed_image is not None and model:
            # Make prediction
            prediction = model.predict(processed_image)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction  # Confidence Score
            confidence_percentage = confidence * 100  # Convert to percentage

            # Determine result label
            result = 'Cancerous✔️' if prediction > 0.5 else 'Non-Cancerous✖️'

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
                selected_symptoms = st.multiselect("Please select any symptoms you are experiencing🤢:", symptoms)

                # Done button
                if st.button("Done"):
                    # Check how many symptoms are selected
                    if len(selected_symptoms) > 3:
                        st.warning("Even if it isn't cancer according to the model, these symptoms could point to other possible illnesses. Please contact medical support.")
                    elif len(selected_symptoms) == 3:
                        st.warning("These symptoms could possibly point to other diseases as well. Be sure to consult a health provider if they continue to worsen.")
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
uploaded_file = st.sidebar.file_uploader("Upload your image🖼️(JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    temp_filename = f"temp_image.{file_extension}"

    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_and_predict(temp_filename, model, last_conv_layer_name)

# Mobile Capture Option
st.sidebar.header("Take a Picture📸")
photo = st.sidebar.file_uploader("Capture a photo🤳", type=["jpg", "jpeg", "png"])
if photo is not None:
    file_extension = photo.name.split('.')[-1]
    captured_filename = f"captured_image.{file_extension}"

    with open(captured_filename, "wb") as f:
        f.write(photo.getbuffer())

    process_and_predict(captured_filename, model, last_conv_layer_name)

# Clear cache button
if st.button("Clear Cache🗑️"):
    st.cache_data.clear()  # Clear the cache
    st.success("Cache cleared successfully!🎯")

if st.sidebar.button("Show Layer Names🌱"):
    st.write("Layer names in EfficientNetB0:")
    layer_names = print_layer_names()
    st.text("\n".join(layer_names))

# Function to collect feedback
def collect_feedback():
    st.title(":rainbow[Feedback] Form✍️")
    
    # Add a text area for the feedback
    feedback = st.text_area("Please share your feedback to improve this app💕", "", height=150)
    
    # Add a submit button
    if st.button("Submit Feedback📩"):
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
    st.info("Your feedback has been recorded😊.")

# Show the feedback form
collect_feedback()

if __name__ == "__main__":
    main()
