
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import kagglehub

# Download required nltk resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

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
        tokens = nltk.word_tokenize(user_input)
        st.write(tokens)

        st.subheader("2. Part-of-Speech Tagging")
        pos_tags = nltk.pos_tag(tokens)
        st.write(pos_tags)

        st.subheader("3. Named Entity Recognition (NER)")
        ne_tree = nltk.chunk.ne_chunk(pos_tags)
        st.text(ne_tree.pformat())

        st.subheader("4. Sentiment Analysis (VADER)")
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(user_input)
        st.write(sentiment)
