import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import norm
import requests
import re
import streamlit as st
import yfinance as yf

st.header("QuantFlow Trading APP")

class Momentum_strategy:
    def __init__(self,price_data,short_sma_period=5,long_sma_period=21):
        self.price_data=price_data
        self.short_sma_period=short_sma_period
        self.long_sma_period=long_sma_period
        self.price_data['long_sma']=None
        self.price_data['short_sma']=None
        self.price_data['Signal']=None
        self.price_data['Position']=None

    def calculate_sma(self):
        self.price_data['short_sma']=self.price_data['Close'].rolling(window= self.short_sma_period).mean()
        self.price_data['long_sma']=self.price_data['Close'].rolling(window=self.long_sma_period).mean
    
    def generate_signals(self):
        signals=[]
        for i in range(1,len(self.price_data)):
            if self.price_data['short_sma'].iloc[i]>self.price_data['long_sma'].iloc[i] and self.price_data['short_sma'].iloc[i-1]<=self.price_data['long_sma'].iloc[i-1]:signals.append(1)
            elif self.price_data['long_sma'].iloc[i]>self.price_data['short_sma'].iloc[i] and self.price_data['long_sma'].iloc[i-1]<=self.price_data['short_sma'].iloc[i-1]:signals.append(-1) 
            else:
                signals.append(0)
        
        signals = [0] +signals
        self.price_data['Signal'] = signals

    def fetch_nifty():
        try:
            nifty_latest = yf.download('^NSEI', interval = '1m', period = '1d')
            nifty_latest = round(nifty_latest.Close[-1], 1)
            return nifty_latest
        except:
            return 26000




