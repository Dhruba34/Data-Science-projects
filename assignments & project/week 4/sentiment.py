import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import joblib
import pandas as pd
lr=joblib.load('lr.pkl')
vect=joblib.load('vector.pkl')
from tqdm import tqdm
#print("Torch version:", torch.__version__)
class RealDataSentimentAnalyzer:
    def __init__(self, tfidf_vectorizer, lr_model, finbert_model_path="ProsusAI/finbert",ticker='AAPL'):
        """
        Initialize all 3 sentiment analyzers.
        
        Parameters:
        - tfidf_vectorizer: From Week 2
        - lr_model: Logistic Regression from Week 2
        - finbert_model_path: FinBERT path
        """
        # VADER (Week 1)
        self.vader = SentimentIntensityAnalyzer()
        
        # Logistic Regression (Week 2)
        self.vectorizer = tfidf_vectorizer
        self.lr_model = lr_model
        
        # FinBERT (Week 3)
        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(finbert_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.df_news=None
        self.ticker=ticker
    
    def score_headline_vader(self, headline):
        """Score using VADER."""
        scores = self.vader.polarity_scores(headline)
        # Return compound score (-1 to +1)
        return scores['compound']
    
    def score_headline_lr(self, headline):
        """Score using Logistic Regression."""
        features = self.vectorizer.transform([headline])
        proba = self.lr_model.predict_proba(features)[0]
        # Return probability difference: positive - negative
        return proba[2] - proba[0]
    
    def score_headline_finbert(self, headline):
        """Score using FinBERT."""
        inputs = self.tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        # Return probability difference: positive - negative
        return (probs[2] - probs[0]).item()

    def score_all_headlines(self, headlines, batch_size=16):
        """
        Fast sentiment scoring with progress tracking.
        """
        headlines = list(headlines)
        total = len(headlines)
    
        # ---------- VADER (fast) ----------
        vader_scores = [
            self.vader.polarity_scores(h)['compound']
            for h in headlines
        ]
    
        # ---------- Logistic Regression (vectorized) ----------
        X = self.vectorizer.transform(headlines)
        lr_probas = self.lr_model.predict_proba(X)
        lr_scores = lr_probas[:, 2] - lr_probas[:, 0]
    
        # ---------- FinBERT (batched + progress) ----------
        finbert_scores = np.zeros(total)
    
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=total, desc="FinBERT inference") as pbar:
                for i in range(0, total, batch_size):
                    batch = headlines[i:i + batch_size]
    
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    ).to(self.device)
    
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
    
                    scores = (probs[:, 2] - probs[:, 0]).cpu().numpy()
                    finbert_scores[i:i + len(scores)] = scores
    
                    # ✅ progress update
                    pbar.update(len(scores))
    
        # ---------- DataFrame ----------
        df = pd.DataFrame({
            "headline": headlines,
            "vader_sentiment": vader_scores,
            "lr_sentiment": lr_scores,
            "finbert_sentiment": finbert_scores
        })
    
        df["mean_sentiment"] = (
            df["vader_sentiment"]
            + df["lr_sentiment"]
            + df["finbert_sentiment"]
        ) / 3
    
        return df

    def load_financial_news(self, news_csv_path):
        """
        Load financial news dataset from CSV.
        Expected columns: date, headline, (optional) ticker
        """
        print(f"Loading news data from {news_csv_path}...")
        self.df_news = pd.read_csv(news_csv_path)

        # Convert date to datetime, coerce invalid dates to NaT
        self.df_news['date'] = pd.to_datetime(
            self.df_news['date'],
            errors='coerce'
        )

        # Remove rows with invalid date/time
        self.df_news = self.df_news.dropna(subset=['date'])

        # Extract date only
        self.df_news['date'] = self.df_news['date'].dt.date

        # Filter by ticker if column exists
        '''if 'ticker' in self.df_news.columns:
            self.df_news = self.df_news[self.df_news['ticker'] == self.ticker]'''

        self.df_news=self.df_news[['date','title']]

        print(f"Loaded {len(self.df_news)} news articles for {self.ticker}")
        return self.df_news
