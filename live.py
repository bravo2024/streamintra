import streamlit as st
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import time

# Set up Streamlit layout
st.title('Real-Time Stock Price Analysis')
ticker_symbol = st.text_input('Enter Stock Symbol (e.g., AAPL for Apple):')
interval_options = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
selected_interval = st.selectbox('Select Time Interval:', interval_options, index=3)
forecast_periods = st.slider('Select Number of Future Values to Forecast', min_value=1, max_value=50, value=5)

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Stock Price')
ax.set_title('Real-Time Stock Price')

# Function to fetch historical data for the selected interval
def fetch_historical_data(ticker_symbol, interval):
    stock = yf.Ticker(ticker_symbol)
    return stock.history(period='5d', interval=interval)

# Function to train Prophet model and make predictions
def apply_prophet(df, periods):
    model = Prophet()
    #print(df.columns)
    #df['Datetime'] = df['Datetime'].dt.tz_localize(None)

    df = df.reset_index().rename(columns={'Datetime': 'ds', 'Close': 'y'})
    # Remove timezone from 'ds' column
    df['ds'] = df['ds'].dt.tz_localize(None)
    #st.write((df.columns))
    #st.write(df.tail(10))

    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Function to update stock prices with Prophet predictions
def update_stock_prices(ticker_symbol, interval, periods):
    while True:
        
        historical_data = fetch_historical_data(ticker_symbol, interval)
        # Apply Prophet
        forecast = apply_prophet(historical_data, periods)
        # Plot results
        ax.clear()
        ax.plot(historical_data.index, historical_data['Close'], label='Original Price', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='red')
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        # Dynamic y-axis limits based on original and predicted prices
        min_price = min(historical_data['Close'].min(), forecast['yhat'].min())
        max_price = max(historical_data['Close'].max(), forecast['yhat'].max())
        ax.set_ylim(min_price * 0.95, max_price * 1.05)  # Adjust margins for better visualization
        
        # Show plot in Streamlit app
        st.pyplot(fig)
        # Wait for 1 minute before fetching new data
        time.sleep(60)
           
if ticker_symbol:
   update_stock_prices(ticker_symbol, selected_interval, forecast_periods)
