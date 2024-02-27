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


# Function to fetch intraday data for the selected interval
def fetch_intraday_data(ticker_symbol, interval):
    stock = yf.Ticker(ticker_symbol)
    # Adjust the period to '1d' and interval to the selected interval
    return stock.history(period='1d', interval=interval)


# Function to train Prophet model and make predictions
def apply_prophet(df, periods,interval):
    model = Prophet()
    st.write(df.tail(10))
    #print(df.columns)
    #df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    #df = df.rename(columns={df.columns[0]: 'ds'})

    df = df.reset_index().rename(columns={'Datetime': 'ds' ,'Date': 'ds', 'Close': 'y'})
    # Remove timezone from 'ds' column
    df['ds'] = df['ds'].dt.tz_localize(None)
    #st.write((df.columns))
    st.write(df.tail(10))
    
    model.fit(df)
    if interval.endswith('m'):  # If interval is in minutes
        freq = str(int(interval[:-1])) + 'T'  # Convert to minutes
    elif interval.endswith('h'):  # If interval is in hours
        freq = str(int(interval[:-1])) + 'H'  # Convert to hours
    else:  # For other intervals (e.g., days, weeks)
        freq = interval
    
    future = model.make_future_dataframe(periods=periods, freq=freq)  # Specify frequency
    forecast = model.predict(future)
    return forecast

# Function to update stock prices with Prophet predictions
# Function to update stock prices with Prophet predictions
def update_stock_prices(ticker_symbol, interval, periods):
    while True:
        if interval.endswith('m'):  # Check if interval is intraday
            # Fetch intraday data
            historical_data = fetch_intraday_data(ticker_symbol, interval)
        else:
            # Fetch historical data
            historical_data = fetch_historical_data(ticker_symbol, interval)

        #st.write(df.tail(10))
        # Fetch historical data
        #historical_data = fetch_historical_data(ticker_symbol, interval)
        # Apply Prophet
        historical_data.index = historical_data.index.tz_localize(None)
        
        # Apply Prophet
        forecast = apply_prophet(historical_data, periods, interval)
        
        # Ensure timezone consistency for 'ds' column in forecast DataFrame
        forecast['ds'] = forecast['ds'].dt.tz_localize(None)
        
        # Plot results
        ax.clear()
        ax.plot(historical_data.index, historical_data['Close'], label='Original Price', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='red')
        ax.legend(loc='upper left')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.set_title('Real-Time Stock Price')
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
        ax.autoscale(enable=True, axis='both', tight=True)  # Autoscale both axes
        
        # Set x-axis limits based on the combined range of actual and predicted data
        min_time = min(historical_data.index.min(), forecast['ds'].min())
        max_time = max(historical_data.index.max(), forecast['ds'].max())
        ax.set_xlim(min_time, max_time)
        
        # Show plot in Streamlit app
        st.pyplot(fig)
      
        
        # Print actual and forecasted values in a single table
        col1, col2 = st.columns(2)
            with col1:
                st.write("### Actual Data")
                st.write(historical_data)
            with col2:
                st.write("### Forecast Data")
                st.write(forecast)
        time.sleep(60)



           
if ticker_symbol:
   update_stock_prices(ticker_symbol, selected_interval, forecast_periods)
