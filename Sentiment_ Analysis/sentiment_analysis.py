import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import streamlit as st
import pandas as pd


def analyze_speeches(speeches):
    sid = SentimentIntensityAnalyzer()
    results = []
    
    for raw_text, preprocessed_text in speeches:
        sentiment_scores = sid.polarity_scores(preprocessed_text)
        results.append((raw_text, sentiment_scores))

    st.subheader("Sentiment Analysis Results")
    for i, (raw_text, sentiment_scores) in enumerate(results):
        st.write(f"**Speech {i+1}:**")
        st.write(f"Positive: {sentiment_scores['pos'] * 100:.2f}%")
        st.write(f"Negative: {sentiment_scores['neg'] * 100:.2f}%")
        st.write(f"Neutral: {sentiment_scores['neu'] * 100:.2f}%")

        if sentiment_scores['compound'] >= 0.05:
            overall_sentiment = "Positive"
        elif sentiment_scores['compound'] <= -0.05:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        st.write(f"Overall Sentiment: **{overall_sentiment}**")

        plot_sentiment_distribution(sentiment_scores, i + 1)
    
    plot_top_words(speeches)

def plot_sentiment_distribution(sentiment_scores, speech_num):
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [sentiment_scores['pos'], sentiment_scores['neg'], sentiment_scores['neu']]
    explode = (0.1, 0, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')

    st.subheader(f"Sentiment Distribution for Speech {speech_num}")
    st.pyplot(fig1)

def plot_top_words(speeches):
    st.subheader("Top 10 Words Contributing to Sentiments")
    all_words = ' '.join([preprocessed_text for _, preprocessed_text in speeches])
    all_tokens = word_tokenize(all_words)
    fdist = FreqDist(all_tokens)
    top_10_words = fdist.most_common(10)

    words, frequencies = zip(*top_10_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.title("Top 10 Words Contributing to Sentiments")
    st.pyplot(plt)

def compare_speeches(speeches):
    sid = SentimentIntensityAnalyzer()
    results = []

    for raw_text, preprocessed_text in speeches:
        sentiment_scores = sid.polarity_scores(preprocessed_text)
        results.append((raw_text, sentiment_scores))

    comparison_df = pd.DataFrame({
        'Speech': [f"Speech {i+1}" for i in range(len(speeches))],
        'Positive': [result[1]['pos'] * 100 for result in results],
        'Negative': [result[1]['neg'] * 100 for result in results],
        'Neutral': [result[1]['neu'] * 100 for result in results]
    })

    st.subheader("Comparison of Sentiments Across Speeches")
    st.table(comparison_df)

    for i, result in enumerate(results):
        plot_sentiment_distribution(result[1], i + 1)
