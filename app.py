import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functions import url_to_sentiment_analysis, preprocess_text, ticker_sentiment_analysis

# Load model and tokenizer once
@st.cache_resource
def load_model():
    model_path = "./finbert_individual2_sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("StockPulse Sentiment Analysis")
st.markdown("Analyze text or financial news URLs for **positive**, **neutral**, or **negative** sentiment.")

# Sidebar mode
mode = st.sidebar.radio("Choose mode", ["Text Input", "URL Analysis"])

if mode == "URL Analysis":
    st.subheader("URL Sentiment Analysis")
    url = st.text_input("Enter a financial news article URL:")
    if st.button("Analyze URL"):
        if url.strip():
            try:
                results = url_to_sentiment_analysis(url, model, tokenizer)

                if results:
                    st.markdown(f"### Sentiment Results from Article at:\n[{url}]({url})")
                    for i, res in enumerate(results, 1):
                        st.markdown(f"**{i}. Ticker:** `{res.get('ticker', 'N/A')}`")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Sentiment:** {res.get('sentiment', 'N/A')}")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Confidence:** {res.get('confidence', 0):.2%}")
                else:
                    st.warning("No sentiment results found. Check if the article is accessible and contains ticker mentions.")

            except Exception as e:
                st.error(f"Error analyzing URL: {str(e)}")
        else:
            st.warning("Please enter a valid URL.")

elif mode == "Text Input":
    st.subheader("Text Sentiment Prediction")
    user_text = st.text_area("Enter your text here:", height=150)
    if st.button("Analyze"):
        if user_text.strip():
            cleaned_text = preprocess_text(user_text)
            inputs = tokenizer(
                cleaned_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            results = ticker_sentiment_analysis(cleaned_text, model, tokenizer)

            if results:
                st.markdown(f"### Sentiment Results:")
                for i, res in enumerate(results, 1):
                        st.markdown(f"**{i}. Ticker:** `{res.get('ticker', 'N/A')}`")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Sentiment:** {res.get('sentiment', 'N/A')}")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Confidence:** {res.get('confidence', 0):.2%}")
            else:
                st.warning("No sentiment results found. Check if the article is accessible and contains ticker mentions.")
        else:
            st.warning("Please enter some text.")

st.markdown("---")
st.caption("📊 Powered by FinBERT - Fine-tuned for sentiment classification")
