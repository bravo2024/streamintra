from datetime import datetime, timedelta
import streamlit as st
from jugaad_data.nse import NSELive
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to load stock data using jugaad_data
def load_data(symbol, num_days=100):
    n = NSELive()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    quote = n.stock_quote(symbol)
    # Convert the quote dictionary into a DataFrame
    df = pd.DataFrame.from_dict(quote, orient='index').transpose()
    # Convert relevant columns to numeric types
    numeric_cols = ['lastPrice', 'change', 'pChange', 'previousClose', 'open', 'close', 'vwap', 'lowerCP', 'upperCP', 'basePrice']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df

# Function to train Prophet model
def train_model(df):
    # Assuming we only need 'lastPrice' and 'Date' columns for forecasting
    df = df[['lastPrice', 'Date']].rename(columns={'Date': 'ds', 'lastPrice': 'y'})
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
    ax.plot(df['Date'], df['lastPrice'], label='Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Last Price')
    ax.set_title('Actual vs. Predicted Last Prices')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Live Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., HDFC):")
    if symbol:
        df = load_data(symbol)
        if not df.empty:
            model = train_model(df)
            forecast_periods = st.slider("Select Number of Forecast Periods", min_value=1, max_value=365, value=30)
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = predict(model, future)
            display_results(df, forecast)

if __name__ == "__main__":
    main()
