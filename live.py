import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import time

# Set up Streamlit layout
st.title('Real-Time Stock Price Analysis')
ticker_symbol = st.text_input('Enter Stock Symbol (e.g., AAPL for Apple):')

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Stock Price')
ax.set_title('Real-Time Stock Price')

# Function to fetch and update stock prices
def update_stock_prices(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    while True:
        try:
            # Get historical prices for the last trading day with 1-minute interval
            historical_prices = stock.history(period='1d', interval='1m')
            # Extract latest price and time
            latest_price = historical_prices['Close'].iloc[-1]
            latest_time = historical_prices.index[-1].strftime('%H:%M:%S')
            # Update plot
            ax.clear()
            ax.plot(historical_prices.index, historical_prices['Close'], label='Stock Price')
            ax.legend(loc='upper left')
            ax.tick_params(axis='x', rotation=45)
            # Show plot in Streamlit app
            st.pyplot(fig)
            # Display latest price
            st.write(f"Latest Price ({latest_time}): {latest_price}")
            # Wait for 1 minute before fetching new data
            time.sleep(60)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Start fetching and updating stock prices if a valid ticker symbol is provided
if ticker_symbol:
    update_stock_prices(ticker_symbol)
