import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# App title and config
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🫁", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🫁 Chest X-Ray Pneumonia Classifier</h1>",
    unsafe_allow_html=True
)

# Load trained model
model = tf.keras.models.load_model(r'D:\medical-xray-cnn\saved_models\cnn_chest_xray.h5')

# Preprocessing
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app uses a CNN model to classify Chest X-ray images into **Normal** or **Pneumonia**.")

# File upload
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]
        label = "🧪 PNEUMONIA" if prediction > 0.5 else "✅ NORMAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown(f"### Prediction: **{label}**")
        st.markdown("### Confidence:")
        st.progress(float(confidence))

        st.success(f"Confidence: {confidence*100:.2f}%")
