import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
@st.cache_resource
def load_cnn():
    return load_model("cnn_model.keras")

model = load_cnn()

st.title("CNN Image Classifier")
st.header("Upload a picture of any of these 10 category:")
st.subheader("airplane✈")
st.subheader("automobile🚗")
st.subheader("bird🦅")
st.subheader("cat🐈")
st.subheader("deer🦌")
st.subheader("dog🐶")
st.subheader("frog🐸")
st.subheader("horse🐴")
st.subheader("ship🚢")
st.subheader("truck🚚")
# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image (change size to your training size)
    img_size = (224, 224)  
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    prediction = model.predict(img_array)
    st.write("Predicted class index:", np.argmax(prediction))
    st.subheader("This is a " + classes[np.argmax(prediction)])
