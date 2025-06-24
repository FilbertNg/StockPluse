# 📊 StockPulse

StockPulse is an intelligent stock sentiment analysis tool that extracts and analyzes financial sentiment from online articles using fine-tuned language models. It helps traders, researchers, and analysts understand how specific stock tickers are discussed in news content.

---

## 🚀 Features

- 🔍 **URL-based Sentiment Extraction**  
  Input a news article URL — StockPulse scrapes and processes the content automatically.

- 🧠 **NER-powered Ticker Matching**  
  Automatically detects relevant company names and maps them to their stock tickers.

- 💬 **Fine-Tuned Sentiment Classification**  
  Uses a BERT-based model to classify sentiment as Positive, Negative, or Neutral for each detected ticker.

- 📈 **Confidence Score Output**  
  Each sentiment prediction is paired with its associated confidence level.

---

## 🖥️ Tech Stack

- **Python**
- **Hugging Face Transformers**
- **Torch / PyTorch**
- **Newspaper** (for web scraping)
- **Streamlit** (for interactive demo)

---

## 🧪 How It Works

1. **User provides a URL**
2. **The article is scraped and cleaned**
3. **Entities (company names) are extracted**
4. **Mapped to stock tickers**
5. **Sentiment analysis is performed per ticker**
6. **Returns: Ticker, Sentiment, Confidence**

---

## 📂 Example Output

```bash
1. Ticker: AAPL  
   Sentiment: Positive  
   Confidence: 87.32%

2. Ticker: TSLA  
   Sentiment: Negative  
   Confidence: 91.44%
```

---
## 🔗 Resources

- [Financial News with Ticker-Level Sentiment (Kaggle)](https://www.kaggle.com/datasets/rdolphin/financial-news-with-ticker-level-sentiment)  
- [SEC Company Ticker List](https://www.sec.gov/file/company-tickers)

---

## 🌐 Live Demo

🔗 **Streamlit Demo** (coming soon)  
> Placeholder — demo URL will be added here once deployed.

---
## 📚 Acknowledgements

- Pretrained BERT models from Hugging Face 🤗  
- Yahoo Finance for ticker data reference  
- FinBERT and Financial Sentiment resources

---

## 📎 TODO (Future Plans)

- ✅ Improved NER mapping  
- [ ] Add multi-URL batch processing  
- [ ] Deploy on cloud with live news integration  
