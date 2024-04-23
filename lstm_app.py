import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load the LSTM model
loaded_lstm_model = tf.keras.models.load_model("lstm_model.h5")

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences([text])
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=128)
    return padded_sequences

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    prediction = loaded_lstm_model.predict(preprocessed_text)
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    pred = np.argmax(prediction)
    sentiment = sentiment_labels[pred]
    return sentiment

# Streamlit app
st.title("Tweet Sentiment Analysis")

# User input for tweet text
tweet_text = st.text_input("Enter the tweet text:")

# Predict sentiment when submit button is clicked
if st.button("Predict Sentiment"):
    if tweet_text:
        sentiment = predict_sentiment(tweet_text)
        st.write(f"The sentiment of the tweet is: {sentiment}")
    else:
        st.write("Please enter a tweet text.")