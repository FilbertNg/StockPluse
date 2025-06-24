import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
# Reveal this if you haven't downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
import random
from newspaper import Article
import time

label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}


# Load dictionaries 
def load_dictionaries(filename):
    with open(filename, 'rb') as file:
        dicts = pickle.load(file)
    return dicts

# Load dictionaries
title_ticker_dict, ticker_title_dict = load_dictionaries("dictionaries.pkl")


# Path to the saved directory
model_save_path = "./finbert_individual2_sentiment_model"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)


def sentiment_analysis_with_ner(sentences, title, model, tokenizer):
    """
    Perform sentiment analysis on sentences containing the specified title using the fine-tuned model.
    """
    # Filter sentences mentioning the title
    title_sentences = [sentence for sentence in sentences if re.search(rf'\b{title}\b', sentence, re.IGNORECASE)]
    
    if not title_sentences:
        return []

    # Predict sentiment for each sentence
    results = []
    for sentence in title_sentences:
        inputs = tokenizer(
            sentence,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        # Map prediction to sentiment label
        sentiment_label = {v: k for k, v in label_mapping.items()}[prediction]
        results.append({
            "sentence": sentence,
            "sentiment": sentiment_label,
            "confidence": confidence
        })

    return results


def combine_sentiments(sentiments):
    """
    Combine multiple sentiment predictions into a single overall sentiment.
    Uses majority voting.
    """
    if not sentiments:
        return "neutral"  # Default sentiment if no sentences available
    sentiment_counts = Counter(sentiments)
    overall_sentiment = sentiment_counts.most_common(1)[0][0]  # Get the most frequent sentiment
    return overall_sentiment


def ticker_sentiment_analysis(content, model, tokenizer):
    """
    Analyze sentiment for tickers mentioned in the content using the fine-tuned model.
    """
    # Step 1: Find mentioned tickers
    mentioned_titles = [title for title in title_ticker_dict if re.search(rf'\b{re.escape(title)}\b', content, re.IGNORECASE)]

    # Step 2: Split content into sentences
    sentences = sent_tokenize(content)

    # Step 3 & 4: Get sentiment analysis for each ticker
    ticker_list = []
    for title in mentioned_titles:
        ticker_sentiments = {}
        ticker_sentiments['ticker'] = title_ticker_dict[title]
        
        # Get list of dicts: [{'sentence': ..., 'sentiment': ..., 'confidence': ...}, ...]
        analysis_results = sentiment_analysis_with_ner(sentences, title, model, tokenizer)
        
        if not analysis_results:
            continue  # skip if no relevant sentences

        # Extract sentiment labels and confidence scores
        sentiments = [res['sentiment'] for res in analysis_results]
        confidences = [res['confidence'] for res in analysis_results]

        # Combine overall sentiment and average confidence
        overall_sentiment = combine_sentiments(sentiments)
        avg_confidence = sum(confidences) / len(confidences)

        # Store results
        ticker_sentiments['sentiment'] = overall_sentiment
        ticker_sentiments['confidence'] = avg_confidence

        ticker_list.append(ticker_sentiments)
    
    return ticker_list


# Web Scrapping
def get_article_text(url, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed for {url}. Retrying in {wait_time:.2f} seconds.")
            time.sleep(wait_time)
            attempt += 1
    print(f"Failed to fetch article after {retries} attempts for {url}")
    return None


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Remove extra spaces
    text = ' '.join(text.split())
    # Return cleaned text
    return text


def url_to_sentiment_analysis(url, model, tokenizer):
    """
    Convert a URL to ticker-level sentiment analysis.
    1. Fetch and preprocess the article content from the URL.
    2. Perform sentiment analysis for tickers mentioned in the article.
    """
    # Step 1: Fetch article content
    raw_text = get_article_text(url)
    if not raw_text:
        print("Failed to fetch article content.")
        return []

    # Step 2: Preprocess text
    cleaned_text = preprocess_text(raw_text)

    # Step 3: Perform sentiment analysis
    sentiments = ticker_sentiment_analysis(cleaned_text, model, tokenizer)
    
    return sentiments
