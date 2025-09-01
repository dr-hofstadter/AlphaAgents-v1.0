import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from autogen import AssistantAgent, UserProxyAgent
import google.generativeai as genai
import time
import matplotlib.pyplot as plt  # Explicitly import matplotlib

# Load API keys from secrets
gemini_api_key = st.secrets["gemini"]["api_key"]
news_api_key = st.secrets["newsapi"]["api_key"]

# Configure Gemini API
genai.configure(api_key=gemini_api_key)
config_list = [{"model": "gemini-1.5-flash", "api_type": "google", "api_key": gemini_api_key}]

# Initialize agents outside the main loop
valuation_agent = AssistantAgent(
    name="Valuation_Agent",
    system_message="",  # Gemini prepends instructions
    llm_config={"config_list": config_list},
)

user_proxy = UserProxyAgent(name="User", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=1)

# Streamlit UI with session state
st.title("AlphaAgents India: Stock Valuation (MVP with Gemini)")
st.write("Enter an NSE ticker (e.g., RELIANCE.NS) and risk profile. Disclaimer: Educational only, not financial advice.")

# Initialize session state
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.result = None
    st.session_state.last_request_time = 0
    st.session_state.chat_history = []
    st.session_state.data_summary = ""
    st.session_state.prices = None
    st.session_state.last_ticker = None  # Track last used ticker
    st.session_state.last_risk_profile = None  # Track last used risk profile

with st.sidebar:
    ticker = st.text_input("Stock Ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS")
    risk_profile = st.selectbox("Risk Profile", ["Neutral", "Averse"])
    analyze_button = st.button("Analyze Stock")

# Calculate annualized return and volatility
def calculate_metrics(prices):
    daily_returns = prices['Close'].pct_change().dropna()
    cumulative_return = daily_returns.sum()
    annualized_return = ((1 + cumulative_return) ** (252 / len(daily_returns))) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    return annualized_return, volatility

# Function to fetch and update stock data
def update_stock_data(ticker):
    stock = yf.Ticker(ticker)
    prices = stock.history(period="4mo")
    if prices.empty:
        st.error("Invalid ticker or no data available.")
        return None, None
    annualized_return, volatility = calculate_metrics(prices)
    data_summary = f"Stock: {ticker}\nAnnualized Return: {annualized_return:.2%}\nVolatility: {volatility:.2%}"
    return prices, data_summary

# Function to get or update recommendation
def get_recommendation(data_summary, risk_profile):
    if (st.session_state.last_ticker == ticker and 
        st.session_state.last_risk_profile == risk_profile and 
        st.session_state.chat_history):
        return st.session_state.chat_history[-1].get('content', 'No recommendation.')
    with st.spinner("Updating recommendation with Gemini..."):
        system_instruction = "Analyze stock prices and volumes for valuation. Provide a buy/sell recommendation based on annualized return and volatility. Use data provided. Respond only once unless further prompted."
        message = f"{system_instruction}\n\nAnalyze {data_summary} with risk-{risk_profile.lower()} profile. Recommend Buy/Sell."
        try:
            result = user_proxy.initiate_chat(valuation_agent, message=message, max_turns=1)
            st.session_state.result = result
            st.session_state.chat_history = result.chat_history
            st.session_state.last_request_time = time.time()
            st.session_state.last_ticker = ticker
            st.session_state.last_risk_profile = risk_profile
            return st.session_state.chat_history[-1].get('content', 'No recommendation.')
        except Exception as e:
            st.error(f"Error: {e}. Please wait 45 seconds and try again.")
            return None

# Update data when ticker or analyze button changes
if analyze_button or (st.session_state.analyzed and st.session_state.last_ticker != ticker):
    with st.spinner("Fetching stock data..."):
        prices, data_summary = update_stock_data(ticker)
        if prices is not None and data_summary is not None:
            st.session_state.prices = prices
            st.session_state.data_summary = data_summary
            recommendation = get_recommendation(data_summary, risk_profile)
            if recommendation:
                st.session_state.analyzed = True
                st.session_state.last_ticker = ticker
                st.session_state.last_risk_profile = risk_profile

# Display results
if st.session_state.analyzed:
    st.subheader("Valuation Report")
    st.write(st.session_state.data_summary)
    st.write("Recommendation:")
    recommendation = get_recommendation(st.session_state.data_summary, risk_profile)
    if recommendation:
        st.write(recommendation)
    
    st.subheader("Price History")
    if st.session_state.prices is not None:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(st.session_state.prices.index, st.session_state.prices['Close'], label='Close Price')
        plt.title(f"{ticker} Close Price (4 Months)")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.error("No price data available to display.")

# Reset button to allow re-analysis
if st.session_state.analyzed:
    if st.button("Reset Analysis"):
        st.session_state.analyzed = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.data_summary = ""
        st.session_state.prices = None
        st.session_state.last_request_time = 0
        st.session_state.last_ticker = None
        st.session_state.last_risk_profile = None
        st.experimental_rerun()

# Run with: streamlit run app.py