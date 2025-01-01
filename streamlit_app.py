import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

class Strategies:
    def __init__(self, price_data, long_sma=200, short_sma=50, initial_capital=100000, transaction_cost=0.001):
        self.price_data = price_data.copy()
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def calculate_emas(self):
        self.price_data['Short_EMA'] = self.price_data['Close'].ewm(span=self.short_sma, adjust=False).mean()
        self.price_data['Long_EMA'] = self.price_data['Close'].ewm(span=self.long_sma, adjust=False).mean()

    def generate_signals(self):
        self.price_data['Signal'] = 0
        crossover = self.price_data['Short_EMA'] > self.price_data['Long_EMA']
        self.price_data.loc[crossover & ~crossover.shift(1).fillna(False), 'Signal'] = 1
        self.price_data.loc[~crossover & crossover.shift(1).fillna(False), 'Signal'] = -1

    def calculate_net_strategy_returns(self):
        self.price_data['Position'] = self.price_data['Signal'].replace({-1: 0}).ffill().fillna(0)
        self.price_data['Daily_Return'] = self.price_data['Close'].pct_change()
        self.price_data['Net_Strategy_Return'] = (
            self.price_data['Position'].shift(1) * self.price_data['Daily_Return']
            - self.transaction_cost * self.price_data['Signal'].abs()
        ).fillna(0)

    def run_strategy(self):
        self.calculate_emas()
        self.generate_signals()
        self.calculate_net_strategy_returns()

    def plot_results(self):
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Close'], name='Close Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Short_EMA'], name='Short EMA', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Long_EMA'], name='Long EMA', line=dict(color='green', dash='dot')))

        buy_signals = self.price_data[self.price_data['Signal'] == 1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))

        sell_signals = self.price_data[self.price_data['Signal'] == -1]
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend",
            template="plotly_dark"
        )

        return fig

    def evaluate_performance(self):
        if 'Net_Strategy_Return' not in self.price_data or self.price_data.empty:
            return {}

        cumulative_returns = (1 + self.price_data['Net_Strategy_Return']).cumprod()
        annualized_return = cumulative_returns.iloc[-1] ** (252 / len(self.price_data)) - 1
        volatility = self.price_data['Net_Strategy_Return'].std() * np.sqrt(252)
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        market_return = self.price_data['Close'].iloc[-1] / self.price_data['Close'].iloc[0] - 1

        return {
            'Market Return': market_return,
            'Total Return': cumulative_returns.iloc[-1] - 1,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

def fetch_nifty_data(years):
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=years)
    nifty_data = yf.download("^NSEI", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if nifty_data.empty:
        st.error("No data retrieved. Please try again with a smaller date range.")
        return pd.DataFrame()
    return nifty_data.dropna()

import streamlit as st

def main():
    performance = {
        "Market Return": 0.1234,
        "Volatility": 0.2567,
        "Total Return": 0.3456,
        "Sharpe Ratio": 1.23,
        "Annualized Return": 0.4567,
        "Max Drawdown": -0.1234,
        "Example Metric": None  
    }
    st.title("Quantitative Trading Strategy Performance")
    col1, col2, col3 = st.columns(3)
    keys = list(performance.keys())
    values = list(performance.values())

    # Distribute metrics across columns
    for i, col in enumerate([col1, col2, col3]):
        for key, value in zip(keys[i::3], values[i::3]):  
            if isinstance(value, (int, float)): 
                formatted_value = (
                    f"{value:.2%}" if 'Return' in key or key == 'Volatility' else f"{value:.2f}"
                )
            else:
                formatted_value = "N/A" 

            col.metric(key, formatted_value)

if __name__ == "__main__":
    main()
