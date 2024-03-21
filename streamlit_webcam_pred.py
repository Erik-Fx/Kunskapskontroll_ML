# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:34:34 2024

@author: erikf
"""



import streamlit as st
import cv2
import numpy as np
import joblib

# Loading the trained k-NN classifier
knn_clf = joblib.load(r"C:\Users\erikf\ec\Machine Learning\Kunskapskontroll\knn_clf_model.pkl")

# Streamlit app title
st.title("Number Prediction")

# Webcam capture
cap = cv2.VideoCapture(0)

# Writes Webcam Feed
st.write("Webcam Feed")

# Placeholder for the live feed
live_feed_placeholder = st.empty()

while True:
    # Capture frame-by-frame
    ret, image = cap.read()

    # Transforming the image for prediction
    img_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bitwise_not(img_gray)


    # Thresholding
    lower_pixel = 100
    upper_pixel = 130
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if img_gray[i, j] <= lower_pixel:
                img_gray[i, j] = 0
            elif img_gray[i, j] > upper_pixel:
                img_gray[i, j] = 255

    flattened_image = img_gray.flatten()
    reshaped_image = flattened_image.reshape(1, -1)

    # Make prediction using the k-NN
    prediction = knn_clf.predict(reshaped_image)

    # Overlay the prediction on the original image
    cv2.putText(image, f"Prediction: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the live feed with prediction using Streamlit
    live_feed_placeholder.image(image, channels="BGR", use_column_width=True)
     

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()
 