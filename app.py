# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import clean_image, get_prediction, make_results
import pandas as pd
from datetime import datetime

# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from utils import clean_image, get_prediction, make_results
import pandas as pd
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)

# Remove Default Streamlit Menu
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title('🌿 Plant Disease Detection')
st.write("Upload your plant's leaf image to detect if it is healthy or diseased.")

# Sidebar Instructions
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload a clear image of the plant leaf.\n"
                 "2. Wait for the model to analyze the image.\n"
                 "3. View the results and confidence levels.")
st.sidebar.write("Supported image formats: PNG, JPG.")

# File Upload
uploaded_file = st.file_uploader("Choose a Leaf Image", type=["png", "jpg", "jpeg"])

# Load Prediction History
@st.cache_data
def load_history():
    try:
        return pd.read_csv("prediction_history.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Timestamp", "Status", "Prediction", "Confidence"])

history = load_history()

# Save Prediction to History
def save_prediction(status, prediction, confidence):
    global history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame({
        "Timestamp": [timestamp],
        "Status": [status],
        "Prediction": [prediction],
        "Confidence": [confidence]
    })
    # Use pd.concat instead of append
    history = pd.concat([history, new_entry], ignore_index=True)
    history.to_csv("prediction_history.csv", index=False)

# Load Model
@st.cache_resource
def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])

    # Combine into a single model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(path)
    return model

# Load the Model
model = load_model("model.h5")

# If a File is Uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Progress Bar
    progress = st.progress(0)
    progress.progress(20)

    # Clean the image
    image = clean_image(image)
    progress.progress(50)

    # Make Predictions
    predictions, predictions_arr = get_prediction(model, image)
    progress.progress(80)

    # Generate Results
    result = make_results(predictions, predictions_arr)
    progress.progress(100)

    # Display Results
    st.markdown(f"""
        ### Results
        - **Status:** The plant {result['status']}
        - **Confidence:** {result['prediction']}
    """)

    # Save results to history
    save_prediction(result["status"], predictions_arr[0], result["prediction"])

    # Visualization of Confidence Levels
    st.subheader("Prediction Confidence Levels")
    class_names = ["Healthy", "Multiple Diseases", "Rust", "Scab"]
    confidences = predictions[0]

    fig, ax = plt.subplots()
    ax.bar(class_names, confidences, color="skyblue")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Distribution")
    st.pyplot(fig)

# Display Prediction History
st.subheader("Prediction History")
if not history.empty:
    st.dataframe(history)
else:
    st.write("No predictions have been made yet.")
