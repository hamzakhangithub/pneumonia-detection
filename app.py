import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model3 = load_model("model.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def predict_pneumonia(img_array):
    predictions = model3.predict(img_array)
    return predictions[0, 0]

def main():
    st.title("Pneumonia Detection with X-ray Images")
    st.text("Upload an X-ray image to check for pneumonia")

    
    uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

    if uploaded_file is not None:
        
        st.image(uploaded_file, caption="Uploaded X-ray Image.", use_column_width=True)

        
        img_array = preprocess_image(uploaded_file)

        
        prediction = predict_pneumonia(img_array)

        
        st.subheader("Prediction:")
        if prediction > 0.5:
            st.write("The model predicts pneumonia.")
        else:
            st.write("The model predicts no pneumonia.")

if __name__ == "__main__":
    main()
