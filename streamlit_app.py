import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np


class Strategies:
    def __init__(self, price_data, long_sma=21, short_sma=5, initial_capital=100000, transaction_cost=0.001):
        self.price_data = price_data
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def calculate_sma(self):
        self.price_data['Short_SMA'] = self.price_data['Close'].rolling(self.short_sma).mean()
        self.price_data['Long_SMA'] = self.price_data['Close'].rolling(self.long_sma).mean()

    def generate_signals(self):
        signals=[]
        for i in range(1,len(self.price_data)):
            if (self.price_data['Short_SMA'].iloc[i]>self.price_data['Long_SMA'].iloc[i] and self.price_data['Short_SMA'].iloc[i-1]<=self.price_data['Long_SMA'].iloc[i-1]):
                signals.append(1)
            elif (self.price_data['Short_SMA'].iloc[i]<self.price_data['Long_SMA'].iloc[i] and self.price_data['Short_SMA'].iloc[i-1]>=self.price_data['Long_SMA'].iloc[i-1]):
                signals.append(-1)
            else:
                signals.append(0)

        signals = [0]+signals
        self.price_data['Signal'] = signals

    def calculate_net_strategy_returns(self):
        self.price_data['Position'] = self.price_data['Signal'].replace({-1: 0}).ffill()
        self.price_data['Daily_Return'] = self.price_data['Close'].pct_change()
        self.price_data['Net_Strategy_Return'] = self.price_data['Position'] * self.price_data['Daily_Return']

    def run_strategy(self):
        self.calculate_sma()
        self.generate_signals()
        self.calculate_net_strategy_returns()

    def plot_results(self):
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Close'], name='Close Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Short_SMA'], name='Short SMA', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Long_SMA'], name='Long SMA', line=dict(color='green', dash='dot')))

        buy_signals = self.price_data[self.price_data['Signal'] == 1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal',marker=dict(color='green', size=10, symbol='triangle-up')))

        sell_signals = self.price_data[self.price_data['Signal'] == -1]
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal',marker=dict(color='red', size=10, symbol='triangle-down')))

        fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_dark"
    )

        return fig


    def evaluate_performance(self):
        cumulative_returns = (1 + self.price_data['Net_Strategy_Return']).cumprod()
        annualized_return = cumulative_returns.iloc[-1] ** (252 / len(self.price_data)) - 1
        volatility = self.price_data['Net_Strategy_Return'].std() * np.sqrt(252)
        max_drawdown = cumulative_returns.div(cumulative_returns.cummax()).min() - 1
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        start_price = self.price_data['Close'].iloc[0]
        end_price = self.price_data['Close'].iloc[-1]
        market_return = (end_price - start_price) / start_price

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
    start_date = end_date - timedelta(days=years * 365)
    nifty_data = yf.download("^NSEI", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if nifty_data.index.tz is not None:
        nifty_data.index = nifty_data.index.tz_localize(None)
    return nifty_data.dropna()

def main():
    st.title("Dual SMA Trading Strategy & Signals")
    st.sidebar.header("Strategy & Parameters")
    st.sidebar.selectbox("Pick a strategy", ['Dual SMA', "Momentum strategy", 'Mean Reversion'])

    years = st.sidebar.number_input("Years of Historical Data", min_value=1, max_value=20, value=5, step=1)
    transaction_cost = st.sidebar.number_input("Transaction cost", min_value=0.0, value=0.001, step=0.0001)
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=100000, step=1000)
    short_window = st.sidebar.number_input("Short SMA Window", min_value=1, max_value=100, value=5)
    long_window = st.sidebar.number_input("Long SMA Window", min_value=1, max_value=365, value=21)
    price_data = fetch_nifty_data(years)

    if not price_data.empty:
        strategy = Strategies(price_data, long_sma=long_window, short_sma=short_window,initial_capital=initial_capital, transaction_cost=transaction_cost)
        strategy.run_strategy()

        with st.container():
            fig = strategy.plot_results()
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Failed to fetch data. Please check your internet connection or ticker symbol.")

if __name__ == "__main__":
    main()
