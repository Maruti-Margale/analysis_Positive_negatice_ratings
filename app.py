import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd # Used only for display purposes in VADER results

# --- App Configuration ---
st.set_page_config(
    page_title="Dual Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- VADER Setup ---

# Download VADER lexicon once
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# --- Hugging Face Setup ---

# Use st.cache_resource to load the potentially heavy model only once
@st.cache_resource
def load_hf_pipeline():
    """Loads the Hugging Face sentiment analysis pipeline."""
    try:
        # We assume dependencies (transformers, torch, etc.) are installed via requirements.txt
        from transformers import pipeline
        # Using a standard, fast sentiment analysis model
        hf_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1 # -1 for CPU, 0 for GPU
        )
        return hf_analyzer, None
    except ImportError as e:
        # Catch specific module import errors
        if 'transformers' in str(e) or 'torch' in str(e):
            return None, "ModuleNotFoundError: The 'transformers' or 'torch' library is missing. Please ensure your 'requirements.txt' includes:\nstreamlit, nltk, transformers, torch, torchaudio, torchdata"
        else:
            return None, f"An unexpected ImportError occurred: {e}"
    except Exception as e:
        # Catch other loading errors (e.g., model download failure)
        return None, f"Error loading Hugging Face model: {e}"

# Load the pipeline and check for errors
hf_analyzer, hf_error = load_hf_pipeline()


# --- Main App Logic ---

st.title("📝 Dual Sentiment Analyzer")
st.markdown("Analyze text sentiment using two different models: **VADER** (Lexicon-based) and **Hugging Face Transformers** (Model-based).")

# Text Input Area
input_text = st.text_area(
    "Paste or type the text/review you want to analyze:",
    value="This product is absolutely amazing! I highly recommend it, but the delivery was a bit slow.",
    height=150
)

st.header("Analysis Results")

if input_text:
    # --- 1. VADER Analysis ---
    st.subheader("1. VADER (Valence Aware Dictionary and sEntiment Reasoner)")
    st.markdown("VADER provides scores for *Positive (pos)*, *Negative (neg)*, *Neutral (neu)*, and a *Compound* score for overall intensity.")

    vader_scores = sia.polarity_scores(input_text)
    st.code(f"Scores: {vader_scores}")

    # Visualize VADER Compound Score
    compound_score = vader_scores['compound']
    st.markdown(f"**Compound Score**")
    st.metric(label="Overall Sentiment", value=f"{compound_score:.3f}")

    # Display individual scores
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{vader_scores['pos']:.3f}")
    col2.metric("Neutral", f"{vader_scores['neu']:.3f}")
    col3.metric("Negative", f"{vader_scores['neg']:.3f}")

    st.markdown("---")

    # --- 2. Hugging Face Analysis ---
    st.subheader("2. Hugging Face Transformers Model")

    if hf_analyzer:
        # Model is loaded, run analysis
        try:
            hf_result = hf_analyzer(input_text)[0]
            label = hf_result['label'].capitalize()
            score = hf_result['score']

            st.success(f"**Sentiment Label:** {label}")
            st.metric(label="Confidence Score", value=f"{score:.4f}")

            st.info(f"The model predicts the sentiment is **{label}** with {score:.2%} confidence.")

        except Exception as e:
            st.error(f"An error occurred during Hugging Face analysis: {e}")
    else:
        # Model failed to load
        st.error("Hugging Face model failed to load.")
        st.warning(hf_error)
else:
    st.info("Please enter text above to begin the sentiment analysis.")

# --- NLTK and Model Notes for deployment clarity ---
st.caption("Note: NLTK downloads required VADER data on first run if not present. Hugging Face model is loaded once on application startup using st.cache_resource.")
