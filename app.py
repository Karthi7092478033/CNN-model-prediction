import tensorflow as tf 
import streamlit as st 
from PIL import Image 
import numpy as np 

#define the class names 
class_names = ['buildings','forests','glacier','mountain','sea','street']

#load your trained model 
model = tf.keras.models.load_model('model.h5')


st.title('CNN Model Prediction')

uploaded_file = st.file_uploader("Choose an Image....",type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='uploaded image .',use_column_width=True)
    st.write("")
    st.write("classifying...")
    
    #preprocessing the image to fit your model input requirments.
    
    img= image.resize((228,228))
    img_array = np.array(img)
    img_array = img_array /255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    
    #make a prediction
    predictions= model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name= class_names[predicted_class_index]
    
    st.write(f"predicted class: {predicted_class_name}")
    