import streamlit as st
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import joblib

# Function to get the latest two working days from Yahoo Finance
def get_recent_data(ticker, selected_date):
    end_date = selected_date.strftime('%Y-%m-%d')
    start_date = (selected_date - pd.DateOffset(days=35)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Function to calculate VWAP
def calculate_vwap(row):
    if row['Volume'] == 0 or pd.isna(row['Volume']):
        return 0
    return row['Close']  # Simplified VWAP calculation for the example

# Load your pre-trained model
def load_model():
    return joblib.load("lgb_model_aug18.pkl")

# UI Setup
st.set_page_config(page_title="Stock Prediction", page_icon="📈", layout="centered")
st.title("Stock Prediction Model")

st.sidebar.header("Input Parameters")
day_open = st.sidebar.number_input('day_open', value=22000)
open_val = st.sidebar.number_input('Open', value=22000)
current_high = st.sidebar.number_input('Current_High', value=22000)
current_low = st.sidebar.number_input('Current_Low', value=22000)
volume_1 = st.sidebar.number_input('Volume_1', value=200000)

# Selectors for date and time
st.sidebar.header("Select Date and Time")
current_time = datetime.now()
month = st.sidebar.selectbox('Month', list(range(1, 13)), index=current_time.month - 1)
day = st.sidebar.selectbox('Day', list(range(1, 32)), index=current_time.day - 1)
year = st.sidebar.selectbox('Year', list(range(2000, current_time.year + 1)), index=current_time.year - 2000)
hour = st.sidebar.selectbox('Hour', list(range(24)), index=current_time.hour)
minute = st.sidebar.number_input('Minute', value=20)

# Construct the selected datetime
selected_date = datetime(year, month, day, hour, minute)

# Retrieve the latest two rows of data from Yahoo Finance
ticker = '^NSEI'
recent_data = get_recent_data(ticker, selected_date)

# Ensure data is available
if len(recent_data) < 2:
    st.error("Insufficient data from Yahoo Finance. Please try again later.")
else:
    # Replace the last value in the recent_data Volume column with the manual input
    recent_data['Volume'].iloc[-1] = volume_1

    # Calculate VWAP
    recent_data['VWAP'] = recent_data.apply(calculate_vwap, axis=1)

    # Calculate moving averages with windows adjusted to trading days
    recent_data['1D_Volume_MA'] = recent_data['Volume'].rolling(window=1, min_periods=1).mean().fillna(0)
    recent_data['2D_Volume_MA'] = recent_data['Volume'].rolling(window=2, min_periods=1).mean().fillna(0)
    recent_data['3D_Volume_MA'] = recent_data['Volume'].rolling(window=3, min_periods=1).mean().fillna(0)
    recent_data['5D_Volume_MA'] = recent_data['Volume'].rolling(window=5, min_periods=1).mean().fillna(0)
    recent_data['7D_Volume_MA'] = recent_data['Volume'].rolling(window=7, min_periods=1).mean().fillna(0)
    
    # Extract necessary fields from recent data
    high_1 = recent_data.iloc[-1]['High']
    low_1 = recent_data.iloc[-1]['Low']
    high_2 = recent_data.iloc[-2]['High']
    low_2 = recent_data.iloc[-2]['Low']
    volume_2 = recent_data.iloc[-2]['Volume']

    # Calculate Fibonacci levels and differences
    fib_ratios = [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]
    df = pd.DataFrame({
        'High_1': [high_1], 'Low_1': [low_1], 'High_2': [high_2], 'Low_2': [low_2],
        'Current_High': [current_high], 'Current_Low': [current_low]
    })
    for high, low in [('High_1', 'Low_1'), ('High_1', 'Low_2'), ('High_2', 'Low_1'), ('High_2', 'Low_2'),
                      ('Low_1', 'High_1'), ('Low_1', 'High_2'), ('Low_2', 'High_1'), ('Low_2', 'High_2'),
                      ('Current_High', 'Current_Low'), ('Current_Low', 'Current_High')]:
        for ratio in fib_ratios:
            df[f'Fib_{ratio}_{high}_{low}'] = df[high] - (df[high] - df[low]) * ratio

    # Calculate differences
    column_pairs = [('Current_High', 'High_1'), ('Current_High', 'High_2'), ('Current_Low', 'Low_1'),
                    ('Current_Low', 'Low_2'), ('High_1', 'High_2'), ('Low_1', 'Low_2')]
    for col1, col2 in column_pairs:
        df[f'Diff_{col1}_{col2}'] = df[col1] - df[col2]

    # Calculate all features
    data = {
        'open': open_val,
        'Month': month,
        'Hour': hour,
        'Minute': minute,
        'day_open': day_open,
        'Current_High': current_high,
        'Current_Low': current_low,
        'High_1': high_1,
        'Low_1': low_1,
        'Volume_1': volume_1,
        'High_2': high_2,
        'Low_2': low_2,
        'Volume_2': volume_2,
        'Diff_Current_High_High_1': current_high - high_1,
        'Diff_Current_High_High_2': current_high - high_2,
        'Diff_Current_Low_Low_1': current_low - low_1,
        'Diff_Current_Low_Low_2': current_low - low_2,
        'Diff_High_1_High_2': high_1 - high_2,
        'Diff_Low_1_Low_2': low_1 - low_2,
    }
    for high, low in [('High_1', 'Low_1'), ('High_1', 'Low_2'), ('High_2', 'Low_1'), ('High_2', 'Low_2'),
                      ('Low_1', 'High_1'), ('Low_1', 'High_2'), ('Low_2', 'High_1'), ('Low_2', 'High_2'),
                      ('Current_High', 'Current_Low'), ('Current_Low', 'Current_High')]:
        for ratio in fib_ratios:
            data[f'Fib_{ratio}_{high}_{low}'] = df[f'Fib_{ratio}_{high}_{low}'].iloc[0]
    for col1, col2 in column_pairs:
        data[f'Diff_{col1}_{col2}'] = df[f'Diff_{col1}_{col2}'].iloc[0]

    # Add more features
    additional_features = {
        'Diff_open_day_open': open_val - day_open,
        'Diff_open_Current_High': open_val - current_high,
        'Diff_open_Current_Low': open_val - current_low,
        'Diff_open_High_1': open_val - high_1,
        'Diff_open_Low_1': open_val - low_1,
        'Diff_open_Volume_1': open_val - volume_1,
        'Diff_open_High_2': open_val - high_2,
        'Diff_open_Low_2': open_val - low_2,
        'Diff_open_Volume_2': open_val - volume_2,
        'Diff_open_Diff_Current_High_High_1': open_val - (current_high - high_1),
        'Diff_open_Diff_Current_High_High_2': open_val - (current_high - high_2),
        'Diff_open_Diff_Current_Low_Low_1': open_val - (current_low - low_1),
        'Diff_open_Diff_Current_Low_Low_2': open_val - (current_low - low_2),
        'Diff_open_Diff_High_1_High_2': open_val - (high_1 - high_2),
        'Diff_open_Diff_Low_1_Low_2': open_val - (low_1 - low_2),
        'Volume_Difference': volume_1 - volume_2,
        'Volume_Percentage_Change': ((volume_1 - volume_2) / volume_2) * 100 if volume_2 != 0 else 0,
        'Volume_Ratio': volume_1 / volume_2 if volume_2 != 0 else 0,
        'Volume_Sum': volume_1 + volume_2,
        'High_Low_Difference': current_high - current_low,
        'VWAP': recent_data['VWAP'].iloc[-1],
        '3D_Volume_MA': recent_data['3D_Volume_MA'].iloc[-1],
        '5D_Volume_MA': recent_data['5D_Volume_MA'].iloc[-1],
        '7D_Volume_MA': recent_data['7D_Volume_MA'].iloc[-1],
        '1D_Volume_MA': recent_data['1D_Volume_MA'].iloc[-1],
        '2D_Volume_MA': recent_data['2D_Volume_MA'].iloc[-1],
    }
    data.update(additional_features)

    # Convert data to DataFrame for prediction
    data_df = pd.DataFrame([data])

    # Display the aggregated row
    st.subheader("Aggregated Data for Model Prediction")
    st.dataframe(data_df)

    # Load the model
    model = load_model()

    prediction = model.predict(data_df)
    prediction_proba = model.predict_proba(data_df)

    st.subheader("Model Prediction")
    st.write("Prediction:", prediction[0])
    st.write("Prediction Probability:", prediction_proba)

    st.markdown("---")
    st.write("**Note:** Ensure that the retrieved data is correct and there are no missing values before proceeding with model prediction.")
