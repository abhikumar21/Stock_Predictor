# Import necessary libraries
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# Define application constants
START_DATE = "2015-01-01"
TODAY_DATE = date.today().strftime("%Y-%m-%d")

# Set up Streamlit interface
st.title("Stock Price Forecast Application")
st.markdown("Use this app to forecast stock prices over a selected number of years.")

# Define stock choices
available_stocks = {
    'Google (GOOG)': 'GOOG',
    'Apple (AAPL)': 'AAPL',
    'Microsoft (MSFT)': 'MSFT',
    'GameStop (GME)': 'GME'
}

# Select stock for prediction
stock_label = st.selectbox("Choose a stock for prediction", list(available_stocks.keys()))
selected_stock = available_stocks[stock_label]

# User selects number of years for prediction
forecast_years = st.slider("Select the number of years to predict:", min_value=1, max_value=4)
forecast_period = forecast_years * 365

# Load stock data function with caching to improve performance
@st.cache
def get_stock_data(ticker):
    stock_data = yf.download(ticker, START_DATE, TODAY_DATE)
    stock_data.reset_index(inplace=True)
    return stock_data

# Display loading text
loading_text = st.text("Fetching stock data...")
data = get_stock_data(selected_stock)
loading_text.text("Data loaded successfully!")

# Display raw data to the user
st.subheader("Most Recent Stock Data")
st.dataframe(data.tail())

# Function to plot the raw stock data
def plot_stock_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Opening Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Closing Price'))
    fig.update_layout(title=f"{stock_label} - Opening and Closing Prices", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Plot the raw data
plot_stock_data()

# Prepare data for forecasting using Prophet
st.subheader(f"Stock Forecast for {forecast_years} Year(s)")
forecast_data = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Initialize and fit the Prophet model
model = Prophet()
model.fit(forecast_data)

# Create future dates and predict using the model
future_dates = model.make_future_dataframe(periods=forecast_period)
predicted_forecast = model.predict(future_dates)

# Display forecast results
st.subheader("Predicted Stock Data")
st.write(predicted_forecast.tail())

# Plot forecast results using Plotly
st.write(f"Forecast plot for the next {forecast_years} year(s):")
forecast_plot = plot_plotly(model, predicted_forecast)
st.plotly_chart(forecast_plot)

# Display the forecast components (trend, seasonality, etc.)
st.write("Detailed Forecast Components")
components_plot = model.plot_components(predicted_forecast)
st.write(components_plot)
