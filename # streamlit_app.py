import numpy as np
cd "c:/Users/Krishna Mohan/Desktop/signature detection/signature detection"

import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image  # Import this

# Load model
from tensorflow.keras.models import load_model

model = load_model("model/signature_model.h5", compile=False)

st.title("Signature Verification")
st.write("Upload a signature image to verify if it's genuine or forged.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
streamlit run streamlit_app.py
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Signature', use_column_width=True)
    st.write("Processing...")

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        st.success("✅ Genuine Signature")
    else:
        st.error("❌ Forged Signature")
