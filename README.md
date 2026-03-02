# Spam-Msg-Detection
A Machine Learning-based SMS Spam Classifier built with TF-IDF and Naive Bayes, deployed using Streamlit.

## Description
SMS Spam Classifier is a Machine Learning-based web application that detects whether a given SMS message is spam or legitimate (ham). The model is trained using **TF-IDF vectorization** and the **Multinomial Naive Bayes** algorithm.  

The project includes text preprocessing (handling URLs, numbers, punctuation, and stopwords) and is deployed through **Streamlit**, allowing real-time testing of messages.


## Technologies Used
- Python 3.x  
- Scikit-learn  
- NLTK  
- Pandas  
- Streamlit  

## How to Run

1. **Clone the repository**
```bash
git clone <your-repo-link>
cd sms-spam-classifier-mai

2.
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

3.
pip install -r requirements.txt
# OR manually
pip install streamlit pandas scikit-learn nltk

4.
python
>>> import nltk
>>> nltk.download('stopwords')
>>> exit()

5.
streamlit run app.py
