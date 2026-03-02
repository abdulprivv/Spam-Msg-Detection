# app.py

import streamlit as st
import pickle
import re

# Preprocessing function (same as in train.py)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
    text = re.sub(r'\d+', 'NUMBER', text)
    text = re.sub(r'[!?.]{2,}', ' PUNCTUATION ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Load model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Model or vectorizer file not found. Please run train.py first.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="SMS Spam Detector", layout="centered")
st.title("📩 SMS Spam Detector")

input_sms = st.text_area("✉️ Enter the message to Detect:")

if st.button("🚀 Predict"):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        # Preprocess and predict
        transformed_sms = preprocess_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Output
        if result == 1:
            st.error("🚫 Spam detected!")
        else:
            st.success("✅ This is a Ham (Not Spam) message.")
