from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import time

# Function to load stock data using Yahoo Finance
def load_data(symbol, interval='1d', num_periods=1):
    # Fetch data from Yahoo Finance
    df = yf.download(symbol, period='1d', interval=interval)
    
    return df

# Function to preprocess data for Prophet model
def preprocess_data(df):
    # Reset index to make 'Datetime' a column
    df.reset_index(inplace=True)
    # Rename columns to match Prophet requirements
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    # Select only required columns
    df = df[['ds', 'y']]
    
    return df

# Function to train Prophet model
def train_model(df):
    model = Prophet()
    model.fit(df)
    return model

# Function to make predictions with Prophet model
def predict(model, future):
    forecast = model.predict(future)
    return forecast

# Function to display results
def display_results(df, forecast):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['ds'], df['y'], label='Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Actual vs. Predicted Prices')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Live Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple Inc.):")
    interval = st.selectbox("Select Interval", ['1d', '1wk', '1mo', '15m', '1h', '4h'])
    num_periods = st.slider("Select Number of Periods", min_value=1, max_value=12, value=1)
    
    if symbol:
        df = load_data(symbol, interval, num_periods)
        if not df.empty:
            df = preprocess_data(df)
            model = train_model(df)
            future = model.make_future_dataframe(periods=30, freq='D')  # Forecast 30 days ahead
            
            # Display initial data and forecast
            forecast = predict(model, future)
            display_results(df, forecast)
            
            # Get the latest historical prices for the stock
            while True:
                # Get the historical prices for the stock
                historical_prices = yf.download(symbol, period='1d', interval=interval)
                
                # Get the latest price and time
                latest_price = historical_prices['Close'].iloc[-1]
                latest_time = historical_prices.index[-1].strftime('%H:%M:%S')
                
                # Clear the plot and plot the new data
                plt.clf()
                plt.plot(historical_prices.index, historical_prices['Close'], label='Stock Value')
                plt.xlabel('Time')
                plt.ylabel('Stock Value')
                plt.title('Stock Value Over Time')
                plt.legend(loc='upper left')
                plt.xticks(rotation=45)
                
                # Update the plot in the Streamlit app
                st.pyplot()
                
                # Show the latest stock value in the app
                st.write(f"Latest Price ({latest_time}): {latest_price}")
                
                # Sleep for 1 minute before fetching new data
                time.sleep(60)

if __name__ == "__main__":
    main()
