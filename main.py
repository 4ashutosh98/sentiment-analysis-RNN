# Step 1: Import Libraries and Load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the IMDB dataset word index
word_index = imdb.get_word_index()
word_index = {v: k for k, v in word_index.items()}

## Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper functions
## Function to decode the reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(word - 3, "<unknown>") for word in sample_review])

## Function to preprocess the user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen= 500)
    return padded_review

# Step 3: Prediction
## Prediction function
def prediction_function(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction[0][0]

## Streamlit App
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

# User Input
review = st.text_input("Enter a movie review:")

if st.button('Classify'):
    # Make prediction
    sentiment, prediction = prediction_function(review)

    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction:.2f}")

else:
    st.write("Please enter a movie review and click the 'Classify' button.")