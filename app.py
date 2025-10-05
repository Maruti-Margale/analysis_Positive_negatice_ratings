import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# --- Configuration and Caching ---

# 1. Cache the NLTK data downloads and model initializations.
# This ensures NLTK setup and model loading only happens once, improving app performance.
@st.cache_resource
def load_nltk_data():
    """Download necessary NLTK data."""
    try:
        # Only VADER is strictly necessary for this app, but including others for robustness
        nltk.download('vader_lexicon', quiet=True)
        # Assuming user might want to expand to POS tagging later
        # nltk.download('punkt', quiet=True)
        # nltk.download('averaged_perceptron_tagger', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"Error loading NLTK data: {e}")
        return None

@st.cache_resource
def load_hf_pipeline():
    """Load the Hugging Face sentiment analysis pipeline."""
    try:
        # Load the default sentiment-analysis model (usually 'distilbert-base-uncased-finetuned-sst-2-english')
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Error loading Hugging Face model. Do you have the 'transformers' library installed? {e}")
        return None

# Load resources
sia = load_nltk_data()
hf_analyzer = load_hf_pipeline()

# --- Analysis Functions ---

def analyze_vader(text, analyzer):
    """Perform VADER sentiment analysis."""
    if not analyzer:
        return "N/A"
    return analyzer.polarity_scores(text)

def analyze_hf(text, analyzer):
    """Perform Hugging Face sentiment analysis."""
    if not analyzer:
        return "N/A"
    return analyzer(text)[0]

# --- Streamlit UI ---

st.set_page_config(page_title="Dual Sentiment Analyzer", layout="centered")

st.title("📝 Dual Sentiment Analyzer")
st.markdown("Analyze text sentiment using two different models: **VADER** (Lexicon-based) and **Hugging Face Transformers** (Model-based).")

# Text input from user
user_input = st.text_area(
    "Paste or type the text/review you want to analyze:",
    "This product is absolutely amazing! I highly recommend it, but the delivery was a bit slow.",
    height=150
)

# Button to trigger analysis
if st.button("Analyze Sentiment", type="primary") and user_input:
    st.subheader("Analysis Results")
    st.divider()

    # 1. VADER Analysis (Lexicon-Based)
    with st.container():
        st.markdown("### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)")
        st.info("VADER provides scores for **Positive (pos)**, **Negative (neg)**, **Neutral (neu)**, and a **Compound** score for overall intensity.")
        vader_scores = analyze_vader(user_input, sia)

        if isinstance(vader_scores, dict):
            st.code(f"Scores: {vader_scores}", language="json")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Compound Score", f"{vader_scores['compound']:.3f}", help="Normalized, weighted composite score. Typically > 0.05 is Positive, < -0.05 is Negative.")
            col2.metric("Positive", f"{vader_scores['pos']:.3f}")
            col3.metric("Neutral", f"{vader_scores['neu']:.3f}")
            col4.metric("Negative", f"{vader_scores['neg']:.3f}")
        else:
            st.error("VADER model failed to load.")

    st.divider()

    # 2. Hugging Face Transformers Analysis (Model-Based)
    with st.container():
        st.markdown("### 2. Hugging Face Transformers Model")
        st.info("A pre-trained deep learning model provides a **Label** (POSITIVE/NEGATIVE) and a **Confidence Score**.")

        hf_result = analyze_hf(user_input, hf_analyzer)

        if isinstance(hf_result, dict):
            label = hf_result.get('label', 'N/A')
            score = hf_result.get('score', 0.0)

            # Determine color based on sentiment label
            color = "green" if label == "POSITIVE" else "red" if label == "NEGATIVE" else "gray"

            st.markdown(
                f"""
                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {color};'>
                    <h4 style='margin-top:0;'>Predicted Sentiment: <span style='color: {color};'>{label}</span></h4>
                    <p style='margin-bottom:0;'>Confidence Score: <strong>{score:.4f}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Hugging Face model failed to load.")

st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
    }
    .stCode {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
