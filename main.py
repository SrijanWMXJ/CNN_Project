import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import tensorflow as tf
from tensorflow import keras


from util import classify, set_background

set_background('./aerial-cardiac-care-checkup.jpg')

# set title
st.title('Pneumonia Detection')

# set header
st.header('Please upload the chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Define the shareable link to the model file on Google Drive
model_url = 'https://drive.google.com/file/d/1-f_df8CiA1jeQygRRUANmJXWBqR6asNn/view?usp=drive_link'

# Download the model file using gdown
model_path = 'classifier_model.h5'
gdown.download(model_url, model_path, quiet=False)

# Load the model
model = keras.models.load_model(model_path)
# load classifier
#model = load_model('./classifier_model.h5')

# load class names
with open('./labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB').resize((150,150))
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
