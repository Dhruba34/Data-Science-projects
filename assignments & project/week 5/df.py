import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
class RealDataPipeline:
    def __init__(self, ticker="AAPL", start_date="2023-01-01", end_date="2024-12-31"):
        """Initialize data pipeline."""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df_prices = None
        self.df_news = None
        self.df_combined = None
    
    def download_stock_prices(self):
        """
        Download historical stock prices using yfinance.
        Returns: DataFrame with OHLCV data
        """
        print(f"Downloading {self.ticker} prices from {self.start_date} to {self.end_date}...")
        self.df_prices = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        # Reset index to make Date a column
        self.df_prices = self.df_prices.reset_index()
        self.df_prices.rename(columns={'Date': 'date'}, inplace=True)
        self.df_prices['date'] = pd.to_datetime(self.df_prices['date']).dt.date
        # 🔥 FIX: Flatten MultiIndex columns if present
        if isinstance(self.df_prices.columns, pd.MultiIndex):
            self.df_prices.columns = [
                col[0] if col[0] != 'date' else 'date'
                for col in self.df_prices.columns
            ]

        
        print(f"Downloaded {len(self.df_prices)} days of price data")
        return self.df_prices
    
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

    
    def align_data(self):
        """
        Align news and prices by date.
        Aggregate multiple news articles per day.
        """
        # Group news by date (multiple headlines per day)
        df_news_daily = self.df_news.groupby('date')['title'].apply(
            lambda x: ' '.join(x)  # Combine headlines for same day
        ).reset_index()
        df_news_daily.rename(columns={'title': 'all_headlines'}, inplace=True)
        #print(self.df_prices.index,df_news_daily.index)
        #print(self.df_prices)
        #print(df_news_daily)
        # Merge prices and news
        self.df_combined = self.df_prices.merge(
            df_news_daily,
            on='date',
            how='inner'
        )
        
        print(f"Combined dataset shape: {self.df_combined.shape}")
        print(f"Date range: {self.df_combined['date'].min()} to {self.df_combined['date'].max()}")
        return self.df_combined
    
    def data_quality_check(self):
        """
        Validate data quality and report issues.
        """
        print("\nDATA QUALITY REPORT")
        print("="*60)
        
        # Missing values
        print(f"Missing Close prices: {self.df_combined['Close'].isna().sum()}")
        print(f"Missing news: {self.df_combined['all_headlines'].isna().sum()}")
        
        # Price statistics
        print(f"\nPrice Statistics:")
        print(f"  Min: ${self.df_combined['Close'].min():.2f}")
        print(f"  Max: ${self.df_combined['Close'].max():.2f}")
        print(f"  Mean: ${self.df_combined['Close'].mean():.2f}")
        print(f"  Volatility: {self.df_combined['Close'].pct_change().std():.4f}")
        
        # News statistics
        print(f"\nNews Statistics:")
        print(f"  Total headlines: {len(self.df_combined)}")
        print(f"  Days covered: {self.df_combined['date'].nunique()}")
        print(f"  Avg words per day: {self.df_combined['all_headlines'].str.split().str.len().mean():.0f}")
