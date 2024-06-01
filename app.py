import streamlit as st
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import joblib
import lightgbm

# Override yfinance with pandas_datareader
yf.pdr_override()

# Function to get the latest two working days from Yahoo Finance
def get_recent_data(ticker, selected_date):
    end_date = selected_date.strftime('%Y-%m-%d')
    start_date = (selected_date - pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    return df.tail(2)

# Function to calculate Fibonacci levels
def calculate_fibonacci_levels(high, low):
    return {
        'Fib_0.5': high - (high - low) * 0.5,
        'Fib_0.618': high - (high - low) * 0.618,
        'Fib_1.5': high -  (high - low) * 1.5,
        'Fib_1.618': high -  (high - low) * 1.618
    }

# Load your pre-trained model
def load_model():
    return joblib.load("lgb_model.pkl")

# UI Setup
st.set_page_config(page_title="Stock Prediction", page_icon="📈", layout="centered")
st.title("Stock Prediction Model")

st.sidebar.header("Input Parameters")
open_val = st.sidebar.number_input('Open', value=22000)
val_1 = st.sidebar.number_input('1', value=22000)
val_2 = st.sidebar.number_input('2', value=22000)
val_3 = st.sidebar.number_input('3', value=22000)
val_4 = st.sidebar.number_input('4', value=22000)
volume_1 = st.sidebar.number_input('Volume_1', value=200000)

# Selectors for date and time
st.sidebar.header("Select Date and Time")
current_time = datetime.now()
month = st.sidebar.selectbox('Month', list(range(1, 13)), index=current_time.month - 1)
day = st.sidebar.selectbox('Day', list(range(1, 32)), index=current_time.day - 1)
year = st.sidebar.selectbox('Year', list(range(2000, current_time.year + 1)), index=current_time.year - 2000)
hour = st.sidebar.selectbox('Hour', list(range(24)), index=current_time.hour)
minute = st.sidebar.selectbox('Minute', list(range(60)), index=current_time.minute)

# Construct the selected datetime
selected_date = datetime(year, month, day, hour, minute)

# Retrieve the latest two rows of data from Yahoo Finance
ticker = '^NSEI'
recent_data = get_recent_data(ticker, selected_date)

# Ensure data is available
if len(recent_data) < 2:
    st.error("Insufficient data from Yahoo Finance. Please try again later.")
else:
    # Calculate necessary fields from recent data
    high_1 = recent_data.iloc[-1]['High']
    low_1 = recent_data.iloc[-1]['Low']
    high_2 = recent_data.iloc[-2]['High']
    low_2 = recent_data.iloc[-2]['Low']
    volume_2 = recent_data.iloc[-2]['Volume']

    fib_levels_11 = calculate_fibonacci_levels(high_1, low_1)
    fib_levels_12 = calculate_fibonacci_levels(high_1, low_2)
    fib_levels_21 = calculate_fibonacci_levels(high_2, low_1)
    fib_levels_22 = calculate_fibonacci_levels(high_2, low_2)

    # Calculate differences
    diff_open_1 = abs(open_val - val_1)
    diff_open_2 = abs(open_val - val_2)
    diff_open_3 = abs(open_val - val_3)
    diff_open_4 = abs(open_val - val_4)

    diff_open_high_1 = abs(open_val - high_1)
    diff_open_low_1 = abs(open_val - low_1)
    diff_open_high_2 = abs(open_val - high_2)
    diff_open_low_2 = abs(open_val - low_2)

    diff_open_fib_0_5_h1_l1 = abs(open_val - fib_levels_11['Fib_0.5'])
    diff_open_fib_0_618_h1_l1 = abs(open_val - fib_levels_11['Fib_0.618'])
    diff_open_fib_1_5_h1_l1 = abs(open_val - fib_levels_11['Fib_1.5'])
    diff_open_fib_1_618_h1_l1 = abs(open_val - fib_levels_11['Fib_1.618'])

    diff_open_fib_0_5_h1_l2 = abs(open_val - fib_levels_12['Fib_0.5'])
    diff_open_fib_0_618_h1_l2 = abs(open_val - fib_levels_12['Fib_0.618'])
    diff_open_fib_1_5_h1_l2 = abs(open_val - fib_levels_12['Fib_1.5'])
    diff_open_fib_1_618_h1_l2 = abs(open_val - fib_levels_12['Fib_1.618'])

    diff_open_fib_0_5_h2_l1 = abs(open_val - fib_levels_21['Fib_0.5'])
    diff_open_fib_0_618_h2_l1 = abs(open_val - fib_levels_21['Fib_0.618'])
    diff_open_fib_1_5_h2_l1 = abs(open_val - fib_levels_21['Fib_1.5'])
    diff_open_fib_1_618_h2_l1 = abs(open_val - fib_levels_21['Fib_1.618'])

    diff_open_fib_0_5_h2_l2 = abs(open_val - fib_levels_22['Fib_0.5'])
    diff_open_fib_0_618_h2_l2 = abs(open_val - fib_levels_22['Fib_0.618'])
    diff_open_fib_1_5_h2_l2 = abs(open_val - fib_levels_22['Fib_1.5'])
    diff_open_fib_1_618_h2_l2 = abs(open_val - fib_levels_22['Fib_1.618'])

    # Create a single row DataFrame for model prediction
    data = {
        'open': open_val,
        '1': val_1,
        '2': val_2,
        '3': val_3,
        '4': val_4,
        'Month': month,
        'Day': day,
        'Year': year,
        'Hour': hour,
        'Minute': minute,
        'High_1': high_1,
        'Low_1': low_1,
        'Volume_1': volume_1,
        'High_2': high_2,
        'Low_2': low_2,
        'Volume_2': volume_2,
        'Fib_0.5_H1_L1': fib_levels_11['Fib_0.5'],
        'Fib_0.5_H1_L2': fib_levels_12['Fib_0.5'],
        'Fib_0.5_H2_L1': fib_levels_21['Fib_0.5'],
        'Fib_0.5_H2_L2': fib_levels_22['Fib_0.5'],
        'Fib_0.618_H1_L1': fib_levels_11['Fib_0.618'],
        'Fib_0.618_H1_L2': fib_levels_12['Fib_0.618'],
        'Fib_0.618_H2_L1': fib_levels_21['Fib_0.618'],
        'Fib_0.618_H2_L2': fib_levels_22['Fib_0.618'],
        'Fib_1.5_H1_L1': fib_levels_11['Fib_1.5'],
        'Fib_1.5_H1_L2': fib_levels_12['Fib_1.5'],
        'Fib_1.5_H2_L1': fib_levels_21['Fib_1.5'],
        'Fib_1.5_H2_L2': fib_levels_22['Fib_1.5'],
        'Fib_1.618_H1_L1': fib_levels_11['Fib_1.618'],
        'Fib_1.618_H1_L2': fib_levels_12['Fib_1.618'],
        'Fib_1.618_H2_L1': fib_levels_21['Fib_1.618'],
        'Fib_1.618_H2_L2': fib_levels_22['Fib_1.618'],
        'Diff_open_1': diff_open_1,
        'Diff_open_2': diff_open_2,
        'Diff_open_3': diff_open_3,
        'Diff_open_4': diff_open_4,
        'Diff_open_High_1': diff_open_high_1,
        'Diff_open_Low_1': diff_open_low_1,
        'Diff_open_High_2': diff_open_high_2,
        'Diff_open_Low_2': diff_open_low_2,
        'Diff_open_Fib_0.5_H1_L1': diff_open_fib_0_5_h1_l1,
        'Diff_open_Fib_0.618_H1_L1': diff_open_fib_0_618_h1_l1,
        'Diff_open_Fib_1.5_H1_L1': diff_open_fib_1_5_h1_l1,
        'Diff_open_Fib_1.618_H1_L1': diff_open_fib_1_618_h1_l1,
        'Diff_open_Fib_0.5_H1_L2': diff_open_fib_0_5_h1_l2,
        'Diff_open_Fib_0.618_H1_L2': diff_open_fib_0_618_h1_l2,
        'Diff_open_Fib_1.5_H1_L2': diff_open_fib_1_5_h1_l2,
        'Diff_open_Fib_1.618_H1_L2': diff_open_fib_1_618_h1_l2,
        'Diff_open_Fib_0.5_H2_L1': diff_open_fib_0_5_h2_l1,
        'Diff_open_Fib_0.618_H2_L1': diff_open_fib_0_618_h2_l1,
        'Diff_open_Fib_1.5_H2_L1': diff_open_fib_1_5_h2_l1,
        'Diff_open_Fib_1.618_H2_L1': diff_open_fib_1_618_h2_l1,
        'Diff_open_Fib_0.5_H2_L2': diff_open_fib_0_5_h2_l2,
        'Diff_open_Fib_0.618_H2_L2': diff_open_fib_0_618_h2_l2,
        'Diff_open_Fib_1.5_H2_L2': diff_open_fib_1_5_h2_l2,
        'Diff_open_Fib_1.618_H2_L2': diff_open_fib_1_618_h2_l2,
        #'Volume_Difference': volume_1 - volume_2,
    }

    # Convert to DataFrame
    input_data = pd.DataFrame([data])

    # Display the aggregated row
    st.subheader("Aggregated Data for Model Prediction")
    st.dataframe(input_data)

    # Load the model
    model = load_model()

    # Predict using the model
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Model Prediction")
    st.write("Prediction:", prediction[0])
    st.write("Prediction Probability:", prediction_proba)

    st.markdown("---")
    st.write("**Note:** Ensure that the retrieved data is correct and there are no missing values before proceeding with model prediction.")

