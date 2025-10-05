import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import kagglehub

# Set up custom nltk_data path
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

# App title
st.title("Amazon Food Review Analyzer")

# Load data using kagglehub
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
    df = pd.read_csv(f"{path}/Reviews.csv")
    return df

df = load_data()

# Sidebar for navigation
option = st.sidebar.selectbox("Choose an option", ["View Dataset", "Review Score Distribution", "Analyze Your Review"])

if option == "View Dataset":
    st.header("Dataset Preview")
    st.dataframe(df[['Id', 'ProductId', 'UserId', 'ProfileName', 'Score', 'Time', 'Summary', 'Text']].head(100))

elif option == "Review Score Distribution":
    st.header("Review Score Distribution")
    score_counts = df['Score'].value_counts().sort_index()
    fig, ax = plt.subplots()
    score_counts.plot(kind='bar', ax=ax)
    ax.set_title("Count of Reviews by Star Rating")
    ax.set_xlabel("Review Score")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

elif option == "Analyze Your Review":
    st.header("NLP Analysis of Your Review")

    user_input = st.text_area("Enter a review to analyze", "This is an excellent product with great taste and quality!")

    if user_input:
        st.subheader("1. Tokenization")
        try:
            tokens = nltk.word_tokenize(user_input)
            st.write(tokens)
        except LookupError as e:
            st.error(f"Tokenization failed: {e}")

        st.subheader("2. Part-of-Speech Tagging")
        try:
            pos_tags = nltk.pos_tag(tokens)
            st.write(pos_tags)
        except Exception as e:
            st.error(f"POS tagging failed: {e}")

        st.subheader("3. Named Entity Recognition (NER)")
        try:
            ne_tree = nltk.chunk.ne_chunk(pos_tags)
            st.text(ne_tree.pformat())
        except Exception as e:
            st.error(f"NER failed: {e}")

        st.subheader("4. Sentiment Analysis (VADER)")
        try:
            sid = SentimentIntensityAnalyzer()
            sentiment = sid.polarity_scores(user_input)
            st.write(sentiment)
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")
