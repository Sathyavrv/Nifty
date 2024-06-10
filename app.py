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

# Load your pre-trained model
def load_model():
    return joblib.load("lgb_model_fit_june7.pkl")

def calculate_fibonacci_levels(df):
    fib_ratios = [0.382, 0.5, 0.618, 1.5, 1.618]
    high_low_combinations = [
        ('High_1', 'Low_1'), ('High_1', 'Low_2'), ('High_2', 'Low_1'), ('High_2', 'Low_2'),
        ('Low_1', 'High_1'), ('Low_1', 'High_2'), ('Low_2', 'High_1'), ('Low_2', 'High_2'),
        ('Current_High', 'Current_Low'), ('Current_Low', 'Current_High')
    ]

    for high, low in high_low_combinations:
        for ratio in fib_ratios:
            df[f'Fib_{ratio}_{high}_{low}'] = df[high] - (df[high] - df[low]) * ratio

    return df

def calculate_column_differences(df):
    column_pairs = [
        ('Current_High', 'High_1'), ('Current_High', 'High_2'),
        ('Current_Low', 'Low_1'), ('Current_Low', 'Low_2'),
        ('High_1', 'High_2'), ('Low_1', 'Low_2')
    ]
    for col1, col2 in column_pairs:
        diff_col_name = f'Diff_{col1}_{col2}'
        df[diff_col_name] = df[col1] - df[col2]

    return df

def calculate_all_differences(df, base_columns, diff_columns):
    new_cols = {}
    for base_col in base_columns:
        for col in diff_columns:
            if base_col != col:
                new_cols[f'Diff_{base_col}_{col}'] = df[base_col] - df[col]
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    return df

def calculate_volume_difference(row):
    return row['Volume_1'] - row['Volume_2']

def calculate_percentage_change(row):
    if row['Volume_2'] == 0:
        return 0
    return ((row['Volume_1'] - row['Volume_2']) / row['Volume_2']) * 100

def calculate_volume_ratio(row):
    if row['Volume_2'] == 0:
        return 0
    return row['Volume_1'] / row['Volume_2']

def calculate_volume_sum(row):
    return row['Volume_1'] + row['Volume_2']

def calculate_high_low_difference(row):
    return row['Current_High'] - row['Current_Low']

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
minute = st.sidebar.number_input('minute', value=20)

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

    df = pd.DataFrame({
        'High_1': [high_1], 'Low_1': [low_1], 'High_2': [high_2], 'Low_2': [low_2],
        'Current_High': [current_high], 'Current_Low': [current_low]
    })
    df = calculate_fibonacci_levels(df)
    df = calculate_column_differences(df)

    # Calculate differences for new columns
    base_columns = ['open_val', 'day_open']
    columns_to_diff = ['open_val', 'day_open', 'Current_High', 'Current_Low', 'High_1', 'Low_1', 'Volume_1',
                       'High_2', 'Low_2', 'Volume_2', 'Diff_Current_High_High_1', 'Diff_Current_High_High_2',
                       'Diff_Current_Low_Low_1', 'Diff_Current_Low_Low_2', 'Diff_High_1_High_2', 'Diff_Low_1_Low_2'] + \
                      [f'Fib_{ratio}_High_1_Low_1' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_High_1_Low_2' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_High_2_Low_1' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_High_2_Low_2' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_Low_1_High_1' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_Low_1_High_2' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_Low_2_High_1' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_Low_2_High_2' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_Current_High_Current_Low' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]] + \
                      [f'Fib_{ratio}_Current_Low_Current_High' for ratio in [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]]

    df = calculate_all_differences(df, base_columns, columns_to_diff)

    df['Volume_Difference'] = df.apply(calculate_volume_difference, axis=1)
    df['Volume_Percentage_Change'] = df.apply(calculate_percentage_change, axis=1)
    df['Volume_Ratio'] = df.apply(calculate_volume_ratio, axis=1)
    df['Volume_Sum'] = df.apply(calculate_volume_sum, axis=1)
    df['High_Low_Difference'] = df.apply(calculate_high_low_difference, axis=1)

    # Create a single row DataFrame for model prediction
    data = {
        'open': open_val,
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
        'Fib_0.382_High_1_Low_1': df.loc[0, 'Fib_0.382_High_1_Low_1'],
        'Fib_0.382_High_1_Low_2': df.loc[0, 'Fib_0.382_High_1_Low_2'],
        'Fib_0.382_High_2_Low_1': df.loc[0, 'Fib_0.382_High_2_Low_1'],
        'Fib_0.382_High_2_Low_2': df.loc[0, 'Fib_0.382_High_2_Low_2'],
        'Fib_0.382_Low_1_High_1': df.loc[0, 'Fib_0.382_Low_1_High_1'],
        'Fib_0.382_Low_1_High_2': df.loc[0, 'Fib_0.382_Low_1_High_2'],
        'Fib_0.382_Low_2_High_1': df.loc[0, 'Fib_0.382_Low_2_High_1'],
        'Fib_0.382_Low_2_High_2': df.loc[0, 'Fib_0.382_Low_2_High_2'],
        'Fib_0.382_Current_High_Current_Low': df.loc[0, 'Fib_0.382_Current_High_Current_Low'],
        'Fib_0.382_Current_Low_Current_High': df.loc[0, 'Fib_0.382_Current_Low_Current_High'],
        'Fib_0.5_High_1_Low_1': df.loc[0, 'Fib_0.5_High_1_Low_1'],
        'Fib_0.5_High_1_Low_2': df.loc[0, 'Fib_0.5_High_1_Low_2'],
        'Fib_0.5_High_2_Low_1': df.loc[0, 'Fib_0.5_High_2_Low_1'],
        'Fib_0.5_High_2_Low_2': df.loc[0, 'Fib_0.5_High_2_Low_2'],
        'Fib_0.5_Low_1_High_1': df.loc[0, 'Fib_0.5_Low_1_High_1'],
        'Fib_0.5_Low_1_High_2': df.loc[0, 'Fib_0.5_Low_1_High_2'],
        'Fib_0.5_Low_2_High_1': df.loc[0, 'Fib_0.5_Low_2_High_1'],
        'Fib_0.5_Low_2_High_2': df.loc[0, 'Fib_0.5_Low_2_High_2'],
        'Fib_0.5_Current_High_Current_Low': df.loc[0, 'Fib_0.5_Current_High_Current_Low'],
        'Fib_0.5_Current_Low_Current_High': df.loc[0, 'Fib_0.5_Current_Low_Current_High'],
        'Fib_0.618_High_1_Low_1': df.loc[0, 'Fib_0.618_High_1_Low_1'],
        'Fib_0.618_High_1_Low_2': df.loc[0, 'Fib_0.618_High_1_Low_2'],
        'Fib_0.618_High_2_Low_1': df.loc[0, 'Fib_0.618_High_2_Low_1'],
        'Fib_0.618_High_2_Low_2': df.loc[0, 'Fib_0.618_High_2_Low_2'],
        'Fib_0.618_Low_1_High_1': df.loc[0, 'Fib_0.618_Low_1_High_1'],
        'Fib_0.618_Low_1_High_2': df.loc[0, 'Fib_0.618_Low_1_High_2'],
        'Fib_0.618_Low_2_High_1': df.loc[0, 'Fib_0.618_Low_2_High_1'],
        'Fib_0.618_Low_2_High_2': df.loc[0, 'Fib_0.618_Low_2_High_2'],
        'Fib_0.618_Current_High_Current_Low': df.loc[0, 'Fib_0.618_Current_High_Current_Low'],
        'Fib_0.618_Current_Low_Current_High': df.loc[0, 'Fib_0.618_Current_Low_Current_High'],
        'Fib_1.5_High_1_Low_1': df.loc[0, 'Fib_1.5_High_1_Low_1'],
        'Fib_1.5_High_1_Low_2': df.loc[0, 'Fib_1.5_High_1_Low_2'],
        'Fib_1.5_High_2_Low_1': df.loc[0, 'Fib_1.5_High_2_Low_1'],
        'Fib_1.5_High_2_Low_2': df.loc[0, 'Fib_1.5_High_2_Low_2'],
        'Fib_1.5_Low_1_High_1': df.loc[0, 'Fib_1.5_Low_1_High_1'],
        'Fib_1.5_Low_1_High_2': df.loc[0, 'Fib_1.5_Low_1_High_2'],
        'Fib_1.5_Low_2_High_1': df.loc[0, 'Fib_1.5_Low_2_High_1'],
        'Fib_1.5_Low_2_High_2': df.loc[0, 'Fib_1.5_Low_2_High_2'],
        'Fib_1.5_Current_High_Current_Low': df.loc[0, 'Fib_1.5_Current_High_Current_Low'],
        'Fib_1.5_Current_Low_Current_High': df.loc[0, 'Fib_1.5_Current_Low_Current_High'],
        'Fib_1.618_High_1_Low_1': df.loc[0, 'Fib_1.618_High_1_Low_1'],
        'Fib_1.618_High_1_Low_2': df.loc[0, 'Fib_1.618_High_1_Low_2'],
        'Fib_1.618_High_2_Low_1': df.loc[0, 'Fib_1.618_High_2_Low_1'],
        'Fib_1.618_High_2_Low_2': df.loc[0, 'Fib_1.618_High_2_Low_2'],
        'Fib_1.618_Low_1_High_1': df.loc[0, 'Fib_1.618_Low_1_High_1'],
        'Fib_1.618_Low_1_High_2': df.loc[0, 'Fib_1.618_Low_1_High_2'],
        'Fib_1.618_Low_2_High_1': df.loc[0, 'Fib_1.618_Low_2_High_1'],
        'Fib_1.618_Low_2_High_2': df.loc[0, 'Fib_1.618_Low_2_High_2'],
        'Fib_1.618_Current_High_Current_Low': df.loc[0, 'Fib_1.618_Current_High_Current_Low'],
        'Fib_1.618_Current_Low_Current_High': df.loc[0, 'Fib_1.618_Current_Low_Current_High'],
        'Diff_open_day_open': open_val - day_open,
        'Diff_open_High_1': open_val - high_1,
        'Diff_open_Low_1': open_val - low_1,
        'Diff_open_High_2': open_val - high_2,
        'Diff_open_Low_2': open_val - low_2,
        'Diff_open_Fib_0.382_High_1_Low_1': open_val - df.loc[0, 'Fib_0.382_High_1_Low_1'],
        'Diff_open_Fib_0.5_High_1_Low_1': open_val - df.loc[0, 'Fib_0.5_High_1_Low_1'],
        'Diff_open_Fib_0.618_High_1_Low_1': open_val - df.loc[0, 'Fib_0.618_High_1_Low_1'],
        'Diff_open_Fib_1.5_High_1_Low_1': open_val - df.loc[0, 'Fib_1.5_High_1_Low_1'],
        'Diff_open_Fib_1.618_High_1_Low_1': open_val - df.loc[0, 'Fib_1.618_High_1_Low_1'],
        'Diff_open_Fib_0.382_High_1_Low_2': open_val - df.loc[0, 'Fib_0.382_High_1_Low_2'],
        'Diff_open_Fib_0.5_High_1_Low_2': open_val - df.loc[0, 'Fib_0.5_High_1_Low_2'],
        'Diff_open_Fib_0.618_High_1_Low_2': open_val - df.loc[0, 'Fib_0.618_High_1_Low_2'],
        'Diff_open_Fib_1.5_High_1_Low_2': open_val - df.loc[0, 'Fib_1.5_High_1_Low_2'],
        'Diff_open_Fib_1.618_High_1_Low_2': open_val - df.loc[0, 'Fib_1.618_High_1_Low_2'],
        'Diff_open_Fib_0.382_High_2_Low_1': open_val - df.loc[0, 'Fib_0.382_High_2_Low_1'],
        'Diff_open_Fib_0.5_High_2_Low_1': open_val - df.loc[0, 'Fib_0.5_High_2_Low_1'],
        'Diff_open_Fib_0.618_High_2_Low_1': open_val - df.loc[0, 'Fib_0.618_High_2_Low_1'],
        'Diff_open_Fib_1.5_High_2_Low_1': open_val - df.loc[0, 'Fib_1.5_High_2_Low_1'],
        'Diff_open_Fib_1.618_High_2_Low_1': open_val - df.loc[0, 'Fib_1.618_High_2_Low_1'],
        'Diff_open_Fib_0.382_High_2_Low_2': open_val - df.loc[0, 'Fib_0.382_High_2_Low_2'],
        'Diff_open_Fib_0.5_High_2_Low_2': open_val - df.loc[0, 'Fib_0.5_High_2_Low_2'],
        'Diff_open_Fib_0.618_High_2_Low_2': open_val - df.loc[0, 'Fib_0.618_High_2_Low_2'],
        'Diff_open_Fib_1.5_High_2_Low_2': open_val - df.loc[0, 'Fib_1.5_High_2_Low_2'],
        'Diff_open_Fib_1.618_High_2_Low_2': open_val - df.loc[0, 'Fib_1.618_High_2_Low_2'],
        'Diff_day_open_High_1': day_open - high_1,
        'Diff_day_open_Low_1': day_open - low_1,
        'Diff_day_open_High_2': day_open - high_2,
        'Diff_day_open_Low_2': day_open - low_2,
        'Diff_day_open_Fib_0.382_High_1_Low_1': day_open - df.loc[0, 'Fib_0.382_High_1_Low_1'],
        'Diff_day_open_Fib_0.5_High_1_Low_1': day_open - df.loc[0, 'Fib_0.5_High_1_Low_1'],
        'Diff_day_open_Fib_0.618_High_1_Low_1': day_open - df.loc[0, 'Fib_0.618_High_1_Low_1'],
        'Diff_day_open_Fib_1.5_High_1_Low_1': day_open - df.loc[0, 'Fib_1.5_High_1_Low_1'],
        'Diff_day_open_Fib_1.618_High_1_Low_1': day_open - df.loc[0, 'Fib_1.618_High_1_Low_1'],
        'Diff_day_open_Fib_0.382_High_1_Low_2': day_open - df.loc[0, 'Fib_0.382_High_1_Low_2'],
        'Diff_day_open_Fib_0.5_High_1_Low_2': day_open - df.loc[0, 'Fib_0.5_High_1_Low_2'],
        'Diff_day_open_Fib_0.618_High_1_Low_2': day_open - df.loc[0, 'Fib_0.618_High_1_Low_2'],
        'Diff_day_open_Fib_1.5_High_1_Low_2': day_open - df.loc[0, 'Fib_1.5_High_1_Low_2'],
        'Diff_day_open_Fib_1.618_High_1_Low_2': day_open - df.loc[0, 'Fib_1.618_High_1_Low_2'],
        'Diff_day_open_Fib_0.382_High_2_Low_1': day_open - df.loc[0, 'Fib_0.382_High_2_Low_1'],
        'Diff_day_open_Fib_0.5_High_2_Low_1': day_open - df.loc[0, 'Fib_0.5_High_2_Low_1'],
        'Diff_day_open_Fib_0.618_High_2_Low_1': day_open - df.loc[0, 'Fib_0.618_High_2_Low_1'],
        'Diff_day_open_Fib_1.5_High_2_Low_1': day_open - df.loc[0, 'Fib_1.5_High_2_Low_1'],
        'Diff_day_open_Fib_1.618_High_2_Low_1': day_open - df.loc[0, 'Fib_1.618_High_2_Low_1'],
        'Diff_day_open_Fib_0.382_High_2_Low_2': day_open - df.loc[0, 'Fib_0.382_High_2_Low_2'],
        'Diff_day_open_Fib_0.5_High_2_Low_2': day_open - df.loc[0, 'Fib_0.5_High_2_Low_2'],
        'Diff_day_open_Fib_0.618_High_2_Low_2': day_open - df.loc[0, 'Fib_0.618_High_2_Low_2'],
        'Diff_day_open_Fib_1.5_High_2_Low_2': day_open - df.loc[0, 'Fib_1.5_High_2_Low_2'],
        'Diff_day_open_Fib_1.618_High_2_Low_2': day_open - df.loc[0, 'Fib_1.618_High_2_Low_2'],
        'Diff_open_Current_High': open_val - current_high,
        'Diff_open_Current_Low': open_val - current_low,
        'Diff_open_Fib_current_0.382_H_L': open_val - df.loc[0, 'Fib_0.382_Current_High_Current_Low'],
        'Diff_open_Fib_current_0.5_H_L': open_val - df.loc[0, 'Fib_0.5_Current_High_Current_Low'],
        'Diff_open_Fib_current_0.618_H_L': open_val - df.loc[0, 'Fib_0.618_Current_High_Current_Low'],
        'Diff_open_Fib_current_1.5_H_L': open_val - df.loc[0, 'Fib_1.5_Current_High_Current_Low'],
        'Diff_open_Fib_current_1.618_H_L': open_val - df.loc[0, 'Fib_1.618_Current_High_Current_Low'],
        'Diff_open_Fib_current_0.382_L_H': open_val - df.loc[0, 'Fib_0.382_Current_Low_Current_High'],
        'Diff_open_Fib_current_0.5_L_H': open_val - df.loc[0, 'Fib_0.5_Current_Low_Current_High'],
        'Diff_open_Fib_current_0.618_L_H': open_val - df.loc[0, 'Fib_0.618_Current_Low_Current_High'],
        'Diff_open_Fib_current_1.5_L_H': open_val - df.loc[0, 'Fib_1.5_Current_Low_Current_High'],
        'Diff_open_Fib_current_1.618_L_H': open_val - df.loc[0, 'Fib_1.618_Current_Low_Current_High'],
        'Volume_Difference': volume_1 - volume_2,
        'Volume_Percentage_Change': ((volume_1 - volume_2) / volume_2) * 100,
        'Volume_Ratio': volume_1 / volume_2,
        'Volume_Sum': volume_1 + volume_2,
        'High_Low_Difference': current_high - current_low,
        'Diff_day_open_Current_High': day_open - current_high,
        'Diff_day_open_Current_Low': day_open - current_low
    }

    # Convert data to DataFrame for prediction
    data_df = pd.DataFrame([data])

    # Display the aggregated row
    st.subheader("Aggregated Data for Model Prediction")
    st.dataframe(data_df)

    # Load the model
    model = load_model()

    # Predict using the model
    prediction = model.predict(data_df)

    st.subheader("Model Prediction")
    st.write("Prediction:", prediction[0])

    st.markdown("---")
    st.write("**Note:** Ensure that the retrieved data is correct and there are no missing values before proceeding with model prediction.")
