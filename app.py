import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from autogen import AssistantAgent, UserProxyAgent
import google.generativeai as genai
import time
import matplotlib.pyplot as plt

# Load API keys
gemini_api_key = st.secrets["gemini"]["api_key"]
news_api_key = st.secrets["newsapi"]["api_key"]

# Configure Gemini
genai.configure(api_key=gemini_api_key)
config_list = [{"model": "gemini-1.5-flash", "api_type": "google", "api_key": gemini_api_key}]

# Agents
valuation_agent = AssistantAgent(
    name="Valuation_Agent",
    system_message="",
    llm_config={"config_list": config_list},
)
user_proxy = UserProxyAgent(name="User", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=1)

# Streamlit UI
st.title("AlphaAgents India: Stock Valuation (MVP with Gemini)")
st.write("Enter an NSE ticker (e.g., RELIANCE.NS) and risk profile. Disclaimer: Educational only, not financial advice.")

# Session state
if "recommendations" not in st.session_state:
    st.session_state.recommendations = {}  # cache { (ticker, risk): recommendation }
if "prices" not in st.session_state:
    st.session_state.prices = None

with st.sidebar:
    ticker = st.text_input("Stock Ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS")
    risk_profile = st.selectbox("Risk Profile", ["Neutral", "Averse"])
    analyze_button = st.button("Analyze Stock")

# Metrics calculation
def calculate_metrics(prices):
    daily_returns = prices['Close'].pct_change().dropna()
    cumulative_return = daily_returns.sum()
    annualized_return = ((1 + cumulative_return) ** (252 / len(daily_returns))) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    return annualized_return, volatility

# Fetch stock data
def update_stock_data(ticker):
    stock = yf.Ticker(ticker)
    prices = stock.history(period="4mo")
    if prices.empty:
        return None, None
    annualized_return, volatility = calculate_metrics(prices)
    summary = f"Stock: {ticker}\nAnnualized Return: {annualized_return:.2%}\nVolatility: {volatility:.2%}"
    return prices, summary

# Safe Gemini call with retry
def safe_gemini_call(message, retries=3, delay=5):
    for attempt in range(retries):
        try:
            result = user_proxy.initiate_chat(valuation_agent, message=message, max_turns=1)
            return result.chat_history[-1].get('content', 'No recommendation.')
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = delay * (2 ** attempt)
                st.warning(f"Gemini quota hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    st.error("Gemini quota exceeded. Showing last saved recommendation if available.")
    return None

# Get recommendation (with cache)
def get_recommendation(data_summary, ticker, risk_profile):
    key = (ticker, risk_profile)
    if key in st.session_state.recommendations:
        return st.session_state.recommendations[key]

    system_instruction = (
        "Analyze stock prices and volumes for valuation. "
        "Provide a buy/sell recommendation based on annualized return and volatility. "
        "Respond only once unless further prompted."
    )
    message = f"{system_instruction}\n\nAnalyze {data_summary} with risk-{risk_profile.lower()} profile. Recommend Buy/Sell."

    with st.spinner("Fetching Gemini recommendation..."):
        rec = safe_gemini_call(message)
        if rec:
            st.session_state.recommendations[key] = rec
        return rec

# Main analysis
if analyze_button:
    with st.spinner("Fetching stock data..."):
        prices, summary = update_stock_data(ticker)
        if prices is not None:
            st.session_state.prices = prices
            recommendation = get_recommendation(summary, ticker, risk_profile)

            st.subheader("Valuation Report")
            st.write(summary)
            if recommendation:
                st.write("Recommendation:")
                st.success(recommendation)

            st.subheader("Price History")
            fig = plt.figure(figsize=(10, 5))
            plt.plot(prices.index, prices['Close'], label='Close Price')
            plt.title(f"{ticker} Close Price (4 Months)")
            plt.xlabel("Date")
            plt.ylabel("Price (INR)")
            plt.legend()
            plt.grid(True)
            st.pyplot(fig)
        else:
            st.error("Invalid ticker or no data available.")

# Reset cache
if st.button("Reset Analysis"):
    st.session_state.recommendations.clear()
    st.session_state.prices = None
    st.experimental_rerun()
