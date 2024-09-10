import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from file_utils import load_and_preprocess
from sentiment_analysis import analyze_speeches, compare_speeches
import nltk
import pandas as pd


nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Streamlit UI
st.title("Independence Day Speech Sentiment Analysis")

# Sidebar for page navigation
page = st.sidebar.selectbox("Choose a page", ["Upload & Analyze", "Comparison"])

# Upload files
uploaded_files = st.file_uploader("Upload Speech Files", type=['txt', 'pdf'], accept_multiple_files=True)

if uploaded_files:
    speeches = load_and_preprocess(uploaded_files)

    # Display uploaded texts
    st.subheader("Uploaded Texts")
    for i, (raw_text, _) in enumerate(speeches):
        st.write(f"**Speech {i+1}:**")
        st.write(raw_text)

    if page == "Upload & Analyze":
        analyze_speeches(speeches)

    elif page == "Comparison":
        if len(speeches) != 2:
            st.warning("Please upload exactly two speeches for comparison.")
        else:
            compare_speeches(speeches)
