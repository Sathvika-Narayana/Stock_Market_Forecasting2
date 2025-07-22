import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("📈 Stock Price Forecasting Dashboard")

stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL")
period = st.slider("Forecast Period (days):", 7, 60, 30)

DEBUG = True

if st.button("Forecast"):
    data = yf.download(stock, start='2015-01-01', auto_adjust=True)
    data.reset_index(inplace=True)
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
    if DEBUG:
        st.write("Downloaded columns:", data.columns.tolist())  # Debug line

    date_col = 'Date'
    # Find the correct close column dynamically
    possible_close = [col for col in data.columns if col.startswith('Close')]
    close_col = f'Close_{stock}' if f'Close_{stock}' in possible_close else 'Close'

    if data.empty:
        st.error("No data found for the given stock symbol and date range. Please check your input or try a different symbol.")
    elif date_col not in data.columns or close_col not in data.columns:
        st.error(f"Downloaded data is missing required columns: {set([close_col, date_col]) - set(data.columns)}. Please try another symbol.")
    else:
        try:
            df = data[[date_col, close_col]].rename(columns={date_col: 'ds', close_col: 'y'}).dropna()
        except Exception as e:
            st.error(f"Error preparing data: {e}")
            st.stop()
        if df.empty or not isinstance(df, pd.DataFrame):
            st.error("No data found after cleaning. Please check your input.")
        else:
            try:
                df['ds'] = pd.to_datetime(df['ds'])
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                df = df.dropna()
            except Exception as e:
                st.error(f"Error processing data: {e}")
                st.stop()
            if df.empty:
                st.error("No data found after cleaning. Please check your input.")
            else:
                st.write("📊 Last 5 rows of Data")
                st.write(df.tail())

                model = Prophet(daily_seasonality=True)
                model.fit(df)

                future = model.make_future_dataframe(periods=period)
                forecast = model.predict(future)

                st.write("✅ Forecast Plot:")
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                st.write("✅ Trend & Seasonality Components:")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                st.success(f"{stock} Forecast for next {period} days generated successfully!")
