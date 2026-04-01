import streamlit as st
import joblib
from preprocess import clean_text

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🐦 Twitter Sentiment Analyzer")

tweet = st.text_input("Enter a tweet")

if st.button("Analyze"):
    cleaned = clean_text(tweet)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    st.success(f"Sentiment: {result}")
