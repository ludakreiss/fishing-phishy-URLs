import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models
import feature_extractor
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import xgboost as xgb
from tcn import TCN 
from tensorflow.keras.optimizers import Adam

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

# from tensorflow.python.keras.utils import get_custom_objects
# from tcn import TCN  # Ensure this is the correct import path




# Now load the model
# # import numpy as np
# from tensorflow.python.keras.models import load_model
# # custom_objects = {"TCN": TCN}

# # Run a basic check with dummy data
# dummy_data = np.random.rand(10, 20, 1)  # Example input shape
# prediction = tcn.predict(dummy_data)

# # Check the shape of the prediction and print it to verify everything works
# st.write("Prediction shape:", prediction.shape)
# st.write("Prediction output:", prediction)

def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    r = true_positives / (possible_positives + K.epsilon())
    return 2 * ((p * r) / (p + r + K.epsilon()))

custom_objects = {
    "TCN": TCN,               # Your custom TCN layer class
    "f1": f1      # Your custom metric function
}


# Load architecture from JSON
with open("tcn_architecture.json", "r") as f:
    tcn = models.model_from_json(f.read(), custom_objects=custom_objects)
tcn.load_weights("TCN #2.weights.h5")  # Load weights (must match architecture)

tcn.compile(
    optimizer=Adam(learning_rate = 0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', f1]
)


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

        label = prediction[0]

        st.text(f"Your URL {'smells phishy' if label == 1 else 'does not smell phishy'}")

    elif model == "Temporal Convolutional Network":
        custom_objects = {'TCN': TCN}

        with tf.keras.utils.custom_object_scope(custom_objects):
            tcn = tf.keras.models.load_model("Models/Dataset #2/TCN/TCN #2.h5")
        tcn.load_weights("path/to/weights.h5")

        scaler = joblib.load("Models/Dataset #2/TCN/scaler.joblib")
        url_scaled = scaler.transform(url_df) 
        url_tcn = np.array(url_scaled).reshape(url_scaled.shape[0], url_scaled.shape[1], 1) 
        prediction = tcn.predict(url_tcn)
        st.text(f"Your URL is {'phishy' if prediction[0] > 0.5 else 'safe'}")

    elif model == "XGBoost":
        xgb = joblib.load("Models/Dataset #2/XGBoost/XGBoost #2.joblib")

        prediction = xgb.predict(url_df)
        st.write("Extracted Features:", url_df)
        st.write("Prediction:", prediction)

        label = prediction[0]

        st.text(f"Your URL is {'phishy' if prediction[0] == 1 else 'safe'}")


