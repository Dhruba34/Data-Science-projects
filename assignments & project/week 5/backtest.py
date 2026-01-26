import pandas as pd
import numpy as np
#!wget "https://raw.githubusercontent.com/Dhruba34/Data-Science-projects/refs/heads/main/assignments%20%26%20project/week%204/df.py"
import df
class BacktestEngine:
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        """
        Initialize backtester.
        
        Parameters:
        - initial_capital: Starting amount of money
        - transaction_cost: Percentage cost per trade (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio_value = [initial_capital]
        self.cash = initial_capital
        self.shares = 0
        self.trades = []
    
    def process_signal(self, date, signal, price, sentiment_score):
        """
        Execute trade based on signal.
        
        Signals:
        - 1: Buy signal
        - -1: Sell signal
        - 0: Hold
        """
        if signal == 1 and self.cash > 0:
            # Buy: use all available cash
            transaction_cost_amount = self.cash * self.transaction_cost
            buy_amount = self.cash - transaction_cost_amount
            self.shares = buy_amount / price
            self.cash = 0
            
            self.trades.append({
                'date': date,
                'action': 'BUY',
                'price': price,
                'shares': self.shares,
                'sentiment': sentiment_score
            })
        
        elif signal == -1 and self.shares > 0:
            # Sell: liquidate all shares
            sell_proceeds = self.shares * price
            transaction_cost_amount = sell_proceeds * self.transaction_cost
            self.cash = sell_proceeds - transaction_cost_amount
            self.shares = 0
            
            self.trades.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'proceeds': sell_proceeds - transaction_cost_amount,
                'sentiment': sentiment_score
            })
    
    def calculate_portfolio_value(self, current_price):
        """Calculate current portfolio value (cash + stock holdings)."""
        stock_value = self.shares * current_price
        return self.cash + stock_value
    
    def run_backtest(self, df, signal_column):
        """
        Run full backtest on data.
        
        Parameters:
        - df: DataFrame with date, Close price, and signal column
        - signal_column: Column name containing buy/sell/hold signals
        
        Returns: Results DataFrame with metrics
        """
        daily_values = []
        
        for idx, row in df.iterrows():
            # Process signal
            self.process_signal(
                date=row['date'],
                signal=row[signal_column],
                price=row['Close'],
                sentiment_score=row.get('sentiment', 0)
            )
            
            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(row['Close'])
            daily_values.append({
                'date': row['date'],
                'portfolio_value': portfolio_value,
                'price': row['Close'],
                'cash': self.cash,
                'shares': self.shares
            })
        
        df_results = pd.DataFrame(daily_values)
        return df_results
    
    def calculate_metrics(self, df_results):
        """
        Calculate performance metrics.
        
        Returns: Dictionary with key metrics
        """
        returns = df_results['portfolio_value'].pct_change().dropna()
        time=(df_results['date'].iloc[-1]-df_results['date'].iloc[0]).days
        total_return = (df_results['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        annual_return = (((df_results['portfolio_value'].iloc[-1] / self.initial_capital) ** (252 / time) - 1)) * 100
        
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        sharpe_ratio = (returns.mean() * 252) / (daily_volatility * np.sqrt(252))
        
        # Max drawdown
        cummax = df_results['portfolio_value'].expanding().max()
        drawdown = (df_results['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Win rate (profitable days)
        profitable_days = (returns > 0).sum()
        win_rate = (profitable_days / len(returns)) * 100
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'final_portfolio_value': df_results['portfolio_value'].iloc[-1]
        }
    def summary_report(self, metrics):
        """Print formatted performance report."""
        report = f"""
        BACKTEST SUMMARY REPORT
        ═════════════════════════════════════════════════════════
        
        RETURNS:
          Total Return: {metrics['total_return']:.2f}%
          Annual Return: {metrics['annual_return']:.2f}%
        
        RISK:
          Annual Volatility: {metrics['annual_volatility']:.2f}%
          Maximum Drawdown: {metrics['max_drawdown']:.2f}%
          Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
        
        TRADING:
          Number of Trades: {metrics['num_trades']}
          Win Rate: {metrics['win_rate']:.2f}%
          Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}
        """
        print(report)
        return report
