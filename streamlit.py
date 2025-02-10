import streamlit as st
from PIL import Image
import numpy as np
from utils import clean_image, get_prediction, make_results
import tensorflow as tf

# Loading the Model
model = load_model('model.h5')  # Make sure to use the correct path for your model file

# Title and Description
st.title('Plant Disease Detection')
st.write("Upload an image of the plant's leaf and the system will tell you if it's healthy or diseased.")

# File Upload Section
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# If the user uploads an image, start processing it
if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = clean_image(image)

    # Make predictions
    predictions, predictions_arr = get_prediction(model, image)

    # Get results
    result = make_results(predictions, predictions_arr)

    # Show the results
    st.write(f"The plant is {result['status']} with a {result['prediction']} prediction.")
