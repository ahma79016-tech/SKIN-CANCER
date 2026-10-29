import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Image Classification App",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Image Classification App")
st.write("Apni image upload karein aur prediction dekhein.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

uploaded_file = st.file_uploader("Image choose karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    
    # Image preprocessing (128x128 target size)
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    if img_array.shape[-1] == 4: # RGBA to RGB
        img_array = img_array[:, :, :3]
        
    img_reshape = np.expand_dims(img_array, axis=0) / 255.0 # Normalize
    
    prediction = model.predict(img_reshape)
    
    st.success("Prediction Complete!")
    st.write(f"**Prediction Output:** {prediction}")
