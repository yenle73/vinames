import streamlit as st
import pickle as pkl
import numpy as np
import sklearn

class_list = {'0': 'Female', '1': 'Male'}
input_ec = open('ec_vinames.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('lrc_vinames.pkl', 'rb')
model = pkl.load(input_md)

st.title('Predict Gender Based on Vietnamese Names')

st.header('Enter a name')
text = st.text_area('Enter a name', '')

if text != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([text])
    rs = str((model.predict(feature_vector))[0])
    st.header('Result')
    st.write(class_list[rs])
