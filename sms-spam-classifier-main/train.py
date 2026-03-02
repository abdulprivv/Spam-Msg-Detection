# train.py

import os
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
    text = re.sub(r'\d+', 'NUMBER', text)
    text = re.sub(r'[!?.]{2,}', ' PUNCTUATION ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Define file path
DATA_FILE = "spam.csv"

try:
    # Load dataset
    df = pd.read_csv(DATA_FILE, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Apply preprocessing
    df['text'] = df['text'].apply(preprocess_text)

    # Vectorize text
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.95)
    X = tfidf.fit_transform(df['text'])
    y = df['label']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("🔍 Evaluation Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save model and vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model and vectorizer saved successfully.")

except FileNotFoundError:
    print(f"❌ Error: File '{DATA_FILE}' not found.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
