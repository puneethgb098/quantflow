import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

class Strategies:
    def __init__(self, price_data, long_sma=200, short_sma=50, initial_capital=100000, transaction_cost=0.001):
        self.price_data = price_data
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def calculate_emas(self):
        self.price_data['Short_EMA'] = self.price_data['Close'].ewm(span=self.short_sma, adjust=False).mean()
        self.price_data['Long_EMA'] = self.price_data['Close'].ewm(span=self.long_sma, adjust=False).mean()

    def generate_signals(self):
        signals = [0] * len(self.price_data)
        for i in range(1, len(self.price_data)):
            if (self.price_data['Short_EMA'].iloc[i] > self.price_data['Long_EMA'].iloc[i] and 
                self.price_data['Short_EMA'].iloc[i-1] <= self.price_data['Long_EMA'].iloc[i-1]):
                signals[i] = 1  # Buy signal
            elif (self.price_data['Short_EMA'].iloc[i] < self.price_data['Long_EMA'].iloc[i] and 
                  self.price_data['Short_EMA'].iloc[i-1] >= self.price_data['Long_EMA'].iloc[i-1]):
                signals[i] = -1  # Sell signal

        self.price_data['Signal'] = signals
        
    def calculate_net_strategy_returns(self):
        self.price_data['Position'] = self.price_data['Signal'].replace({-1: 0}).ffill().fillna(0)
        self.price_data['Daily_Return'] = self.price_data['Close'].pct_change()
        self.price_data['Entry_Price'] = np.nan
        self.price_data['Unrealized_Return'] = 0
        net_returns = []

        for i in range(1, len(self.price_data)):
            prev_position = self.price_data['Position'].iloc[i-1]
            curr_position = self.price_data['Position'].iloc[i]
            current_price = self.price_data['Close'].iloc[i]
            entry_price = self.price_data['Entry_Price'].iloc[i-1] if pd.notna(self.price_data['Entry_Price'].iloc[i-1]) else current_price

            if curr_position != 0:
                unrealized_return = (current_price - entry_price) * curr_position
            else:
                unrealized_return = 0

            strategy_return = prev_position * self.price_data['Daily_Return'].iloc[i]

            if curr_position != prev_position:
                strategy_return -= self.transaction_cost * np.abs(curr_position - prev_position)

            if curr_position != prev_position and curr_position != 0:
                self.price_data['Entry_Price'].iloc[i] = current_price
            else:
                self.price_data['Entry_Price'].iloc[i] = entry_price

            net_returns.append(strategy_return + unrealized_return)

        net_returns = [0] + net_returns
        self.price_data['Net_Strategy_Return'] = net_returns

    def run_strategy(self):
        self.calculate_emas()
        self.generate_signals()
        self.calculate_net_strategy_returns()

    def plot_results(self):
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Close'], name='Close Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Short_EMA'], name='Short EMA', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Long_EMA'], name='Long EMA', line=dict(color='green', dash='dot')))

        buy_signals = self.price_data[self.price_data['Signal'] == 1].copy()
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))

        sell_signals = self.price_data[self.price_data['Signal'] == -1].copy()
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend",
            template="plotly_dark"
        )

        return fig

    def evaluate_performance(self):
        if 'Net_Strategy_Return' not in self.price_data or len(self.price_data) == 0:
            return {}

        # Calculate cumulative returns
        self.price_data['Cumulative_Return'] = (1 + self.price_data['Net_Strategy_Return']).cumprod()
        
        cumulative_returns = self.price_data['Cumulative_Return']
        
        annualized_return = (cumulative_returns.iloc[-1] ** (252 / len(self.price_data)) - 1) if len(self.price_data) > 0 else 0
        volatility = self.price_data['Net_Strategy_Return'].std() * np.sqrt(252)
        
        max_drawdown = cumulative_returns.div(cumulative_returns.cummax()).min() - 1 if not cumulative_returns.empty else 0
        
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
    st.title("Advanced Dual EMA Trading Strategy & Signals")
    st.sidebar.header("Strategy & Parameters")
    
    years = st.sidebar.number_input("Years of Historical Data", min_value=1, max_value=20, value=5, step=1)
    transaction_cost = st.sidebar.number_input("Transaction cost", min_value=0.0, value=0.001, step=0.0001)
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=100000, step=1000)
    short_window = st.sidebar.number_input("Short EMA Window", min_value=1, max_value=100, value=50)
    long_window = st.sidebar.number_input("Long EMA Window", min_value=1, max_value=365, value=200)
    
    price_data = fetch_nifty_data(years)

    if not price_data.empty:
        strategy = Strategies(price_data, long_sma=long_window, short_sma=short_window,
                              initial_capital=initial_capital, transaction_cost=transaction_cost)
        
        strategy.run_strategy()

        with st.container():
            fig = strategy.plot_results()
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Evaluate Strategy"):
                performance = strategy.evaluate_performance()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Market Return", f"{performance.get('Market Return', 0):.2%}")
                    st.metric("Volatility", f"{performance.get('Volatility', 0):.2%}")

                with col2:
                    st.metric("Total Return", f"{performance.get('Total Return', 0):.2%}")
                    st.metric("Sharpe Ratio", f"{performance.get('Sharpe Ratio', 0):.2f}")

                with col3:
                    st.metric("Annualized Return", f"{performance.get('Annualized Return', 0):.2%}")
                    st.metric("Max Drawdown", f"{performance.get('Max Drawdown', 0):.2%}")

if __name__ == "__main__":
    main()
