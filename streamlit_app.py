import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Strategies:
    def __init__(self, price_data, long_sma=21, short_sma=5):
        self.price_data = price_data
        self.short_sma = short_sma
        self.long_sma = long_sma

    def calculate_sma(self):
        self.price_data['Short_SMA'] = self.price_data['Close'].rolling(self.short_sma).mean()
        self.price_data['Long_SMA'] = self.price_data['Close'].rolling(self.long_sma).mean()

    def generate_signals(self):
        signals = []
        for i in range(1, len(self.price_data)):
            if (
                self.price_data['Short_SMA'].iloc[i] > self.price_data['Long_SMA'].iloc[i]
                and self.price_data['Short_SMA'].iloc[i - 1] <= self.price_data['Long_SMA'].iloc[i - 1]
            ):
                signals.append(1)  # Buy Signal
            elif (
                self.price_data['Short_SMA'].iloc[i] < self.price_data['Long_SMA'].iloc[i]
                and self.price_data['Short_SMA'].iloc[i - 1] >= self.price_data['Long_SMA'].iloc[i - 1]
            ):
                signals.append(-1)  # Sell Signal
            else:
                signals.append(0)  # Hold Signal

        signals = [0] + signals
        self.price_data['Signal'] = signals

    def run_strategy(self):
        self.calculate_sma()
        self.generate_signals()

    def plot_results(self):
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(x=self.price_data.index, y=self.price_data['Close'], name='Close Price', line=dict(color='blue')))

        fig.add_trace(go.Scatter(x=self.price_data.index,y=self.price_data['Short_SMA'],name='Short SMA',line=dict(color='red', dash='dot')))

        fig.add_trace(go.Scatter(x=self.price_data.index,y=self.price_data['Long_SMA'],name='Long SMA',line=dict(color='green', dash='dot')))

        buy_signals = self.price_data[self.price_data['Signal'] == 1]
        fig.add_trace(go.Scatter(x=buy_signals.index,y=buy_signals['Close'],mode='markers',name='Buy Signal',marker=dict(color='green', size=10, symbol='triangle-up')))

        sell_signals = self.price_data[self.price_data['Signal'] == -1]
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],mode='markers',name='Sell Signal',marker=dict(color='red', size=10, symbol='triangle-down'),))

        fig.update_layout(xaxis_title="Date",yaxis_title="Price",legend_title="Legend",template="plotly_dark",)

        return fig
    
def fetch_nifty_data():
    data = yf.download("^NSEI", period="10y", interval="1d")
    return data


def main():
    st.title("Dual SMA Trading Strategy & Signals")
    st.sidebar.header("Strategy & Parameters")
    st.sidebar.selectbox("Pick a strategy",['Dual SMA',"Momentum strategy",'Mean Reversion'])
    short_window = st.sidebar.number_input("Short SMA Window", min_value=1, max_value=100, value=5)
    long_window = st.sidebar.number_input("Long SMA Window", min_value=1, max_value=365, value=21)
    price_data = fetch_nifty_data()

    if not price_data.empty:
        strategy = Strategies(price_data, long_sma=long_window, short_sma=short_window)
        strategy.run_strategy()

        with st.container(border=True):
            fig = strategy.plot_results()
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Failed to fetch data. Please check your internet connection or ticker symbol.")


if __name__ == "__main__":
    main()
