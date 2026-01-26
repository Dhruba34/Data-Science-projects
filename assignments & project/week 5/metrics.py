import backtest
import pandas as pd
import numpy as np
import df
import sentiment
import joblib
lr=joblib.load('lr.pkl')
vect=joblib.load('vector.pkl')

class metrics_calculate:
    def __init__(self,start='2020-01-01',end='2025-12-31'):
        self.start=start
        self.end=end
    def calculate(self):
        obj2=df.RealDataPipeline(start_date=self.start,end_date=self.end)
        obj2.download_stock_prices()
        obj2.load_financial_news('apple_news_data.csv')
        obj2.align_data()
        #print("The merged dataframe is as follows") 
        obj2.df_combined.head()

        obj=sentiment.RealDataSentimentAnalyzer(vect,lr)
        headlines=obj2.df_combined['all_headlines']
        obj2.df_combined[['vader_sentiment','lr_sentiment','finbert_sentiment']]=obj.score_all_headlines(headlines)[['vader_sentiment','lr_sentiment','finbert_sentiment']]
        data=obj2.df_combined
        data['hold']=0
        data.loc[data.index[-1], 'hold'] = -1
        data.loc[data.index[0], 'hold'] = 1

        def signal_from_sentiment(sentiment):
            threshold=0.02
            if sentiment > threshold:
                return 1  # Buy
            elif sentiment < -threshold:
                return -1  # Sell
            else:
                return 0  # Hold
        data['vader_sentiment']=data['vader_sentiment'].apply(signal_from_sentiment)
        data['lr_sentiment']=data['lr_sentiment'].apply(signal_from_sentiment)
        data['finbert_sentiment']=data['finbert_sentiment'].apply(signal_from_sentiment)
        data.head()


        # Run backtests
        backtester_vader = backtest.BacktestEngine(initial_capital=100000)
        results_vader = backtester_vader.run_backtest(data, 'vader_sentiment')
        metrics_vader = backtester_vader.calculate_metrics(results_vader)

        backtester_lr = backtest.BacktestEngine(initial_capital=100000)
        results_lr = backtester_lr.run_backtest(data, 'lr_sentiment')
        metrics_lr = backtester_lr.calculate_metrics(results_lr)

        backtester_finbert = backtest.BacktestEngine(initial_capital=100000)
        results_finbert = backtester_finbert.run_backtest(data, 'finbert_sentiment')
        metrics_finbert = backtester_finbert.calculate_metrics(results_finbert)

        backtester_hold = backtest.BacktestEngine(initial_capital=100000)
        results_hold = backtester_hold.run_backtest(data, 'hold')
        metrics_hold = backtester_hold.calculate_metrics(results_hold)

        return metrics_vader, metrics_lr, metrics_finbert, metrics_hold,results_vader, results_lr, results_finbert, results_hold
