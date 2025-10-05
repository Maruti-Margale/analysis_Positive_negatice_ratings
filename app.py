import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np # Needed for array processing in batch analysis
import altair as alt # Recommended for interactive charts in Streamlit

# --- App Configuration ---
st.set_page_config(
    page_title="Dual Sentiment Analyzer",
    layout="wide", # Use wide layout for better data visualization
    initial_sidebar_state="collapsed",
)
# Note: Streamlit's default theme (light/dark) is controlled by the user's local settings.
# To enforce a specific look, we'll use a clean layout and control chart colors.

# --- VADER Setup ---

# Download VADER lexicon once
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Function to apply VADER analysis to a DataFrame
@st.cache_data(show_spinner="Analyzing sentiment with VADER...")
def apply_vader_to_df(df, text_column):
    """Calculates VADER scores for every row in a DataFrame."""
    if text_column not in df.columns:
        return pd.DataFrame(), f"Error: Text column '{text_column}' not found in the uploaded file."
    
    # Apply SIA to all text entries
    df['VADER_Scores'] = df[text_column].apply(lambda x: sia.polarity_scores(str(x)))
    
    # Extract compound score into a separate column
    df['VADER_Compound'] = df['VADER_Scores'].apply(lambda x: x['compound'])
    
    return df, None

# --- Hugging Face Setup ---

# Use st.cache_resource to load the potentially heavy model only once
@st.cache_resource
def load_hf_pipeline():
    """Loads the Hugging Face sentiment analysis pipeline."""
    try:
        from transformers import pipeline
        # Using a standard, fast sentiment analysis model
        hf_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1 # -1 for CPU, 0 for GPU
        )
        return hf_analyzer, None
    except ImportError as e:
        if 'transformers' in str(e) or 'torch' in str(e):
            return None, "ModuleNotFoundError: The 'transformers' or 'torch' library is missing. Please ensure your 'requirements.txt' includes:\nstreamlit, nltk, transformers, torch, torchaudio, torchdata"
        else:
            return None, f"An unexpected ImportError occurred: {e}"
    except Exception as e:
        return None, f"Error loading Hugging Face model: {e}"

# Load the pipeline and check for errors
hf_analyzer, hf_error = load_hf_pipeline()


# --- Main App Logic ---

st.title("📝 Dual Sentiment Analyzer")
st.markdown("Analyze text sentiment using two different models: **VADER** (Lexicon-based) and **Hugging Face Transformers** (Model-based).")

# --- Tabs for switching between single analysis and batch analysis ---
tab1, tab2 = st.tabs(["Single Text Analysis", "Dataset Batch Analysis"])

with tab1:
    # Text Input Area
    input_text = st.text_area(
        "Paste or type the text/review you want to analyze:",
        value="This product is absolutely amazing! I highly recommend it, but the delivery was a bit slow.",
        height=150
    )

    st.header("Single Review Results")

    if input_text:
        # --- 1. VADER Analysis ---
        st.subheader("1. VADER (Valence Aware Dictionary and sEntiment Reasoner)")
        st.markdown("VADER provides scores for *Positive (pos)*, *Negative (neg)*, *Neutral (neu)*, and a *Compound* score for overall intensity.")

        vader_scores = sia.polarity_scores(input_text)
        st.code(f"Scores: {vader_scores}")

        # Visualize VADER Compound Score
        compound_score = vader_scores['compound']
        st.metric(label="Overall Sentiment (Compound Score)", value=f"{compound_score:.3f}")

        # Display individual scores
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", f"{vader_scores['pos']:.3f}")
        col2.metric("Neutral", f"{vader_scores['neu']:.3f}")
        col3.metric("Negative", f"{vader_scores['neg']:.3f}")

        st.markdown("---")

        # --- 2. Hugging Face Analysis ---
        st.subheader("2. Hugging Face Transformers Model")

        if hf_analyzer:
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
            st.error("Hugging Face model failed to load.")
            st.warning(hf_error)
    else:
        st.info("Please enter text above to begin the sentiment analysis.")

with tab2:
    st.header("Upload and Analyze Review Dataset")
    st.markdown("Upload a CSV file containing reviews. The app will calculate VADER scores for all reviews and visualize the results.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.dataframe(df.head())
            
            # --- Column Selection ---
            st.subheader("Configuration")
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            score_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number) and df[col].nunique() < 10]
            
            col_text, col_score = st.columns(2)
            
            review_column = col_text.selectbox(
                "Select the column containing the **Review Text**:",
                options=text_cols,
                index=0 if text_cols else None
            )

            rating_column = col_score.selectbox(
                "Select the column containing the **Original Rating** (e.g., 1-5 stars) for grouping:",
                options=['None'] + score_cols,
                index=0
            )

            # --- Run Analysis ---
            if review_column:
                analyzed_df, analysis_error = apply_vader_to_df(df.copy(), review_column)
                
                if analysis_error:
                    st.error(analysis_error)
                elif not analyzed_df.empty:
                    st.subheader("3. VADER Batch Analysis Results")
                    st.info(f"Analysis complete for {len(analyzed_df)} reviews.")
                    
                    # --- Visualization ---
                    if rating_column != 'None':
                        # Calculate the mean compound score per rating
                        chart_data = analyzed_df.groupby(rating_column)['VADER_Compound'].mean().reset_index()
                        chart_data.columns = ['Rating', 'Average Compound Score']
                        
                        st.markdown(f"#### Average VADER Compound Score by {rating_column}")
                        
                        # --- MODIFICATION: Updated Altair color range for better contrast ---
                        chart = alt.Chart(chart_data).mark_bar().encode(
                            x=alt.X('Rating:O', axis=alt.Axis(title=f'Original Rating ({rating_column})')),
                            y=alt.Y('Average Compound Score:Q'),
                            # Using a color scale that transitions clearly from negative (red) to positive (green)
                            color=alt.Color(
                                'Average Compound Score:Q', 
                                scale=alt.Scale(domain=[-1, 0, 1], range=['#ef4444', '#f59e0b', '#10b981']), # Red, Amber, Emerald Green
                            ),
                            tooltip=['Rating', alt.Tooltip('Average Compound Score', format='.3f')]
                        ).properties(
                            height=400
                        ).interactive() # Allows zooming/panning
                        
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning("Select a Rating column to view the chart visualization.")

                    # Show a sample of the analyzed data
                    st.markdown("#### Sample of VADER Results")
                    st.dataframe(analyzed_df[[review_column, 'VADER_Compound']].head(10))
            else:
                st.warning("Please select the Review Text column to run the analysis.")
                
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")


# --- NLTK and Model Notes for deployment clarity ---
st.caption("Note: NLTK downloads required VADER data on first run if not present. Hugging Face model is loaded once on application startup using st.cache_resource.")
