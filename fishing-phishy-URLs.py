import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models, Model
import feature_extractor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
import xgboost as xgb
from tensorflow.keras.optimizers import Adam
from TCN_build import create_model, f1

# Setting up the page components
st.set_page_config(
    page_title = "Phsihing Detection System",
    page_icon=":fish:",
    layout="centered",
)

st.markdown(r"""
    <h1 style='text-align: center;'>How Phishy Is This Website?</h1>
            
    <style>
        .vertical-space { margin-top: 50px; }
    </style>
    <div class="vertical-space"></div>
            
    <h3 style='text-align: center;'> Paste a URL and see how likely is it to be a Phish &#x1F41F;! </h3>
            
    <style>
        .vertical-space { margin-top: 70px; }
    </style>
    <div class="vertical-space"></div>
            
    <div style='text-align: center; font-size: 30px;'>Pick your model</div>""", 
    
    unsafe_allow_html=True)

# Selecting whuich model would the user liek to pick
model = st.selectbox("", 
                     ["Linear SVC", "Multi-layer Perceptron", "Random Forest", "Temporal Convolutional Network", "XGBoost"], 
                     help="Beware, each model that is picked may produce different results!")

st.markdown(r"""
    <style>
        .vertical-space { margin-top: 70px; }
    </style>
    <div class="vertical-space"></div>
            
    <div style='text-align: center; font-size: 30px;'>Enter your URL</div>
            
    """,
    
    unsafe_allow_html=True)

url = st.text_input(label="")

# If the user enters a URL, then the URL is extracted and is fed to the chosen model 
if url:
    url_df = feature_extractor.extract_features(url)
    

    if model == "Linear SVC":
        svc = joblib.load("Models/Dataset #2/Linear SVC/LinearSVC #2.joblib")
        scaler = joblib.load("Models/Dataset #2/Linear SVC/scaler.joblib")

        url_scaled = scaler.transform(url_df) 

        prediction = svc.predict(url_scaled)

        label = prediction[0]
        if label ==1:
            st.markdown(r"<div style='text-align: center; font-size: 20px;'>This URL is most likely a phishing rod! &#x1F3A3</div>", unsafe_allow_html=True)
        elif label == 0:
             st.markdown(r"<div style='text-align: center; font-size: 20px;'>This is a safe URL.</div>", unsafe_allow_html=True)

    elif model == "Multi-layer Perceptron":
        mlp = joblib.load("Models/Dataset #2/MLP/MLP #2.joblib")
        scaler = joblib.load("Models/Dataset #2/MLP/scaler.joblib")

        url_scaled = scaler.transform(url_df) 

        prediction = mlp.predict(url_scaled)

        label = prediction[0]
        if label ==1:
            st.markdown(r"<div style='text-align: center; font-size: 20px;'>This URL is most likely a phishing rod! &#x1F3A3</div>", unsafe_allow_html=True)
        elif label == 0:
             st.markdown(r"<div style='text-align: center; font-size: 20px;'>This is a safe URL.</div>", unsafe_allow_html=True)

    elif model == "Random Forest":
        rf = joblib.load("Models/Dataset #2/Random Forest/rand_forest #2.joblib")
        prediction = rf.predict(url_df)

        label = prediction[0]

        if label ==1:
            st.markdown(r"<div style='text-align: center; font-size: 20px;'>This URL is most likely a phishing rod! &#x1F3A3</div>", unsafe_allow_html=True)
        elif label == 0:
             st.markdown(r"<div style='text-align: center; font-size: 20px;'>This is a safe URL.</div>", unsafe_allow_html=True)

    elif model == "Temporal Convolutional Network":

        model = create_model()
        model.compile(
            optimizer=Adam(learning_rate = 0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', f1]
        )
        model.load_weights("Models/Dataset #2/TCN/TCN #2.weights.h5")

        scaler = joblib.load("Models/Dataset #2/TCN/scaler.joblib")
        url_scaled = scaler.transform(url_df) 
        url_tcn = np.array(url_scaled).reshape(url_scaled.shape[0], url_scaled.shape[1], 1) 
        prediction = model.predict(url_tcn)
        label = label = np.argmax(prediction[0])

        if label ==1:
            st.markdown(r"<div style='text-align: center; font-size: 20px;'>This URL is most likely a phishing rod! &#x1F3A3</div>", unsafe_allow_html=True)
        elif label == 0:
             st.markdown(r"<div style='text-align: center; font-size: 20px;'>This is a safe URL.</div>", unsafe_allow_html=True)


    elif model == "XGBoost":
        xgb = joblib.load("Models/Dataset #2/XGBoost/XGBoost #2.joblib")

        prediction = xgb.predict_proba(url_df)
        probability = prediction[0][1]

        if probability > 0.5:
            st.markdown(r"<div style='text-align: center; font-size: 20px;'>This URL is most likely a phishing rod! &#x1F3A3</div>", unsafe_allow_html=True)
        elif probability <= 0.5:
             st.markdown(r"<div style='text-align: center; font-size: 20px;'>This is a safe URL.</div>", unsafe_allow_html=True)


