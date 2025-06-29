import streamlit as st
import joblib

# Load model and tools
model = joblib.load("random_forest_sentiment.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Optional text cleaning
def preprocess(text):
    return text  # Add preprocessing if needed

# Streamlit app
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title(" sentiment Analysis App")
st.write("Enter product and review text to analyze sentiment.")

# Input fields
product = st.text_input("Product Name")
review = st.text_area("Customer Review")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("❗ Please enter a review.")
    else:
        combined = preprocess(product + " " + review)
        vector = tfidf.transform([combined])
        pred = model.predict(vector)[0]
        sentiment = label_encoder.inverse_transform([pred])[0]
        st.success(f"Predicted Sentiment: **{sentiment}**")



