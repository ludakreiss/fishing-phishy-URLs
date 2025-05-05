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


def main():

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
                
        <div style='text-align: center; font-size: 30px;'>Pick your model</div>
                
        <div style='text-align: center; font-size: 20px;'>Beware that each model may output a different result to the same URL.</div>""", 
        
        unsafe_allow_html=True)

    with st.sidebar:
        st.header("Tips on How to Upload The URL for The Best Results")
        st.sidebar.markdown(r"""
        <ol>
            <li>Make sure that you do not click on the URL if you beleive it is suspicious. 
                    Instead,right-click the URL and click "Copy link address" to avoid opening possible phishing URLs. </li>
            <li>When trying a URL out, make sure that the text field include the explicit version of the URL and not the bare. For example, "google.com" &#x274E
                    but "https://www.google.com" &#x2705; </li>
            
        </ol> """, 
        
        unsafe_allow_html=True)


    # Selecting whuich model would the user liek to pick
    model = st.selectbox("", 
                        ["Linear SVC", "Multi-layer Perceptron", "Random Forest", "Temporal Convolutional Network", "XGBoost"])

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
            svc = joblib.load("Models/Dataset #2/Linear SVC/LinearSVC #2.joblib") #Linear SVC
            scaler = joblib.load("Models/Dataset #2/Linear SVC/scaler.joblib") #Load in its respective scaler

            url_scaled = scaler.transform(url_df) #scale the URL
            prediction = svc.predict_proba(url_scaled) #Predict the label of the URL
            probability = prediction[0][1]

            st.markdown(
                f"<div style='text-align: center; font-size: 20px;'>The chances of this URL being phishy is {probability * 100:.2f}% !</div>",
                unsafe_allow_html=True
            )
            
        elif model == "Multi-layer Perceptron":
            mlp = joblib.load("Models/Dataset #2/MLP/MLP #2.joblib") #Multi-Layer Precptron
            scaler = joblib.load("Models/Dataset #2/MLP/scaler.joblib") #Load in its respective scaler

            url_scaled = scaler.transform(url_df)  #scale the URL
            prediction = mlp.predict_proba(url_df) #Predict the label of the URL
            probability = prediction[0][1]

            st.markdown(
                f"<div style='text-align: center; font-size: 20px;'>The chances of this URL being phishy is {probability * 100:.2f}% !</div>",
                unsafe_allow_html=True
            )
            
        elif model == "Random Forest":
            rf = joblib.load("Models/Dataset #2/Random Forest/rand_forest #2.joblib") #Random Forest
            prediction = rf.predict_proba(url_df) #Predict the label of the URL
            probability = prediction[0][1]

            st.markdown(
                f"<div style='text-align: center; font-size: 20px;'>The chances of this URL being phishy is {probability * 100:.2f}% !</div>",
                unsafe_allow_html=True
            )
            
        elif model == "Temporal Convolutional Network":
            tcn = create_model() #TCN: Build the archtecture of the TCN model
            tcn.compile(
                optimizer=Adam(learning_rate = 0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', f1]
            ) #TCN: Compile the model 
            tcn.load_weights("Models/Dataset #2/TCN/TCN #2.weights.h5") #TCN: Load in the weights of a trained TCN model

            scaler = joblib.load("Models/Dataset #2/TCN/scaler.joblib") #Load in its respective scaler
            url_scaled = scaler.transform(url_df)  #scale the URL
            url_tcn = np.array(url_scaled).reshape(url_scaled.shape[0], url_scaled.shape[1], 1) #Reshape the url_scaled as the model expects shape (15, 1, 1)
            prediction = tcn.predict(url_tcn) #Predict the label of the URL
            probability = prediction[0][1]

            st.markdown(
                f"<div style='text-align: center; font-size: 20px;'>The chances of this URL being phishy is {probability * 100:.2f}% !</div>",
                unsafe_allow_html=True
            )
            
        elif model == "XGBoost":
            xgb = joblib.load("Models/Dataset #2/XGBoost/XGBoost #2.joblib") #XGBoost
            prediction = xgb.predict_proba(url_df)  #Predict the label of the URL
            probability = prediction[0][1]
            
            st.markdown(
                f"<div style='text-align: center; font-size: 20px;'>The chances of this URL being phishy is {probability * 100:.2f}% !</div>",
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()