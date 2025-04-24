import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import feature_extractor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from tcn import TCN 

st.set_page_config(
    page_title = "Phsihing Detection System",
    page_icon=":fish:",
    layout="centered",
)

st.markdown("""
    <h1 style='text-align: center;'>How Phishy Is This Website?</h1>
            
    <style>
        .vertical-space { margin-top: 50px; }
    </style>
    <div class="vertical-space"></div>
            
    <h3 style='text-align: center;'> Paste a URL and see how likely is it to be a Phish &#x1F41F;! </h3>
    """, unsafe_allow_html=True)



model = st.selectbox("Pick the model", 
                     ["Linear SVC", "Multi-layer Perceptron", "Random Forest", "Temporal Convolutional Network", "XGBoost"], 
                     help ="Beware, each model that is picked may produce different results! ")

url = st.text_input(label = "Enter the URL")

scaler = joblib.load("Models/Dataset #2/TCN/scaler.joblib")

get_custom_objects().update({'TCN': TCN})

if url:
    url_df = feature_extractor.extract_features(url)
    
    

    # if model == "Temporal Convolutional Network":
    #     url_input = np.array(url_scaled).reshape(url_scaled.shape[0], url_scaled.shape[1], 1)
    # else:
    #     url_input = pd.DataFrame(url_scaled, columns=url_df.columns)

    if model == "Linear SVC":
        svc = joblib.load("Models/Dataset #2/Linear SVC/LinearSVC #2.joblib")
        scaler = joblib.load("Models/Dataset #2/Linear SVC/scaler.joblib")

        url_scaled = scaler.transform(url_df) 

        prediction = svc.predict(url_scaled)

        label = prediction[0]
        st.text(f"Your URL {'smells phishy' if label == 1 else 'does not smell phishy'}")

    elif model == "Multi-layer Perceptron":
        mlp = joblib.load("Models/Dataset #2/MLP/MLP #2.joblib")
        scaler = joblib.load("Models/Dataset #2/MLP/scaler.joblib")

        url_scaled = scaler.transform(url_df) 

        prediction = mlp.predict(url_scaled)

        label = prediction[0]
        st.text(f"Your URL {'smells phishy' if label == 1 else 'does not smell phishy'}")

    elif model == "Random Forest":
        rf = joblib.load("Models/Dataset #2/Random Forest/rand_forest #2.joblib")

        prediction = rf.predict(url_df)

        st.write("Extracted Features:", url_df)
        st.write("Prediction:", prediction)

        label = prediction[0]

        st.text(f"Your url is {prediction}% phishy")

    elif model == "Temporal Convolutional Network":
        tcn = models.load_model("Models/Dataset #2/TCN/TCN #2.h5")
        scaler = joblib.load("Models/Dataset #2/TCN/scaler.joblib")
        url_scaled = scaler.transform(url_df) 
        url_tcn = np.array(url_scaled).reshape(url_scaled.shape[0], url_scaled.shape[1], 1) 
        prediction = tcn.predict(url_tcn)
        st.text(f"Your url is {prediction}% phishy")

    elif model == "XGBoost":
        xgb = joblib.load("Models/Dataset #2/XGBoost/XGBoost #2.joblib")

        prediction = xgb.predict(url_df)
        st.write("Extracted Features:", url_df)
        st.write("Prediction:", prediction)

        label = prediction[0]

        st.text(f"Your URL is {'phishy' if prediction[0] == 1 else 'safe'}")


