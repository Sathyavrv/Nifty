
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
    return joblib.load("lgb_model_fit_june5.pkl")

def calculate_fibonacci_levels(df):
    fib_ratios = [0.382, 0.5, 0.618, 1.5, 1.618]
    for ratio in fib_ratios:
        df[f'Fib_{ratio}_H1_L1'] = df['High_1'] - (df['High_1'] - df['Low_1']) * ratio
        df[f'Fib_{ratio}_H1_L2'] = df['High_1'] - (df['High_1'] - df['Low_2']) * ratio
        df[f'Fib_{ratio}_H2_L1'] = df['High_2'] - (df['High_2'] - df['Low_1']) * ratio
        df[f'Fib_{ratio}_H2_L2'] = df['High_2'] - (df['High_2'] - df['Low_2']) * ratio
        df[f'Fib_{ratio}_L1_H1'] = df['Low_1'] - (df['Low_1'] - df['High_1']) * ratio
        df[f'Fib_{ratio}_L1_H2'] = df['Low_1'] - (df['Low_1'] - df['High_2']) * ratio
        df[f'Fib_{ratio}_L2_H1'] = df['Low_2'] - (df['Low_2'] - df['High_1']) * ratio
        df[f'Fib_{ratio}_L2_H2'] = df['Low_2'] - (df['Low_2'] - df['High_2']) * ratio
        df[f'Fib_current_{ratio}_H_L'] = df['Current_High'] - (df['Current_High'] - df['Current_Low']) * ratio
        df[f'Fib_current_{ratio}_L_H'] = df['Current_Low'] - (df['Current_Low'] - df['Current_High']) * ratio
    return df

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

    fib_levels_1 = {
        'Fib_0.382': df.loc[0, 'Fib_0.382_H1_L1'], 'Fib_0.5': df.loc[0, 'Fib_0.5_H1_L1'],
        'Fib_0.618': df.loc[0, 'Fib_0.618_H1_L1'], 'Fib_1.5': df.loc[0, 'Fib_1.5_H1_L1'],
        'Fib_1.618': df.loc[0, 'Fib_1.618_H1_L1']
    }
    fib_levels_2 = {
        'Fib_0.382': df.loc[0, 'Fib_0.382_H1_L2'], 'Fib_0.5': df.loc[0, 'Fib_0.5_H1_L2'],
        'Fib_0.618': df.loc[0, 'Fib_0.618_H1_L2'], 'Fib_1.5': df.loc[0, 'Fib_1.5_H1_L2'],
        'Fib_1.618': df.loc[0, 'Fib_1.618_H1_L2']
    }

    fib_levels_l1_h1 = {
        'Fib_0.382': df.loc[0, 'Fib_0.382_L1_H1'], 'Fib_0.5': df.loc[0, 'Fib_0.5_L1_H1'],
        'Fib_0.618': df.loc[0, 'Fib_0.618_L1_H1'], 'Fib_1.5': df.loc[0, 'Fib_1.5_L1_H1'],
        'Fib_1.618': df.loc[0, 'Fib_1.618_L1_H1']
    }
    fib_levels_l1_h2 = {
        'Fib_0.382': df.loc[0, 'Fib_0.382_L1_H2'], 'Fib_0.5': df.loc[0, 'Fib_0.5_L1_H2'],
        'Fib_0.618': df.loc[0, 'Fib_0.618_L1_H2'], 'Fib_1.5': df.loc[0, 'Fib_1.5_L1_H2'],
        'Fib_1.618': df.loc[0, 'Fib_1.618_L1_H2']
    }
    fib_levels_l2_h1 = {
        'Fib_0.382': df.loc[0, 'Fib_0.382_L2_H1'], 'Fib_0.5': df.loc[0, 'Fib_0.5_L2_H1'],
        'Fib_0.618': df.loc[0, 'Fib_0.618_L2_H1'], 'Fib_1.5': df.loc[0, 'Fib_1.5_L2_H1'],
        'Fib_1.618': df.loc[0, 'Fib_1.618_L2_H1']
    }
    fib_levels_l2_h2 = {
        'Fib_0.382': df.loc[0, 'Fib_0.382_L2_H2'], 'Fib_0.5': df.loc[0, 'Fib_0.5_L2_H2'],
        'Fib_0.618': df.loc[0, 'Fib_0.618_L2_H2'], 'Fib_1.5': df.loc[0, 'Fib_1.5_L2_H2'],
        'Fib_1.618': df.loc[0, 'Fib_1.618_L2_H2']
    }



    # Calculate differences
    diff_open_high_1 = open_val - high_1
    diff_open_low_1 = open_val - low_1
    diff_open_high_2 = open_val - high_2
    diff_open_low_2 = open_val - low_2

    diff_open_fib_0_382_h1_l1 = open_val - fib_levels_1['Fib_0.382']
    diff_open_fib_0_5_h1_l1 = open_val - fib_levels_1['Fib_0.5']
    diff_open_fib_0_618_h1_l1 = open_val - fib_levels_1['Fib_0.618']
    diff_open_fib_1_5_h1_l1 = open_val - fib_levels_1['Fib_1.5']
    diff_open_fib_1_618_h1_l1 = open_val - fib_levels_1['Fib_1.618']

    diff_open_fib_0_382_h1_l2 = open_val - fib_levels_2['Fib_0.382']
    diff_open_fib_0_5_h1_l2 = open_val - fib_levels_2['Fib_0.5']
    diff_open_fib_0_618_h1_l2 = open_val - fib_levels_2['Fib_0.618']
    diff_open_fib_1_5_h1_l2 = open_val - fib_levels_2['Fib_1.5']
    diff_open_fib_1_618_h1_l2 = open_val - fib_levels_2['Fib_1.618']

    diff_open_fib_0_382_h2_l1 = open_val - fib_levels_1['Fib_0.382']
    diff_open_fib_0_5_h2_l1 = open_val - fib_levels_1['Fib_0.5']
    diff_open_fib_0_618_h2_l1 = open_val - fib_levels_1['Fib_0.618']
    diff_open_fib_1_5_h2_l1 = open_val - fib_levels_1['Fib_1.5']
    diff_open_fib_1_618_h2_l1 = open_val - fib_levels_1['Fib_1.618']

    diff_open_fib_0_382_h2_l2 = open_val - fib_levels_2['Fib_0.382']
    diff_open_fib_0_5_h2_l2 = open_val - fib_levels_2['Fib_0.5']
    diff_open_fib_0_618_h2_l2 = open_val - fib_levels_2['Fib_0.618']
    diff_open_fib_1_5_h2_l2 = open_val - fib_levels_2['Fib_1.5']
    diff_open_fib_1_618_h2_l2 = open_val - fib_levels_2['Fib_1.618']

    # Differences for day open column
    diff_day_open_high_1 = day_open - high_1
    diff_day_open_low_1 = day_open - low_1
    diff_day_open_high_2 = day_open - high_2
    diff_day_open_low_2 = day_open - low_2

    diff_day_open_fib_0_382_h1_l1 = day_open - fib_levels_1['Fib_0.382']
    diff_day_open_fib_0_5_h1_l1 = day_open - fib_levels_1['Fib_0.5']
    diff_day_open_fib_0_618_h1_l1 = day_open - fib_levels_1['Fib_0.618']
    diff_day_open_fib_1_5_h1_l1 = day_open - fib_levels_1['Fib_1.5']
    diff_day_open_fib_1_618_h1_l1 = day_open - fib_levels_1['Fib_1.618']

    diff_day_open_fib_0_382_h1_l2 = day_open - fib_levels_2['Fib_0.382']
    diff_day_open_fib_0_5_h1_l2 = day_open - fib_levels_2['Fib_0.5']
    diff_day_open_fib_0_618_h1_l2 = day_open - fib_levels_2['Fib_0.618']
    diff_day_open_fib_1_5_h1_l2 = day_open - fib_levels_2['Fib_1.5']
    diff_day_open_fib_1_618_h1_l2 = day_open - fib_levels_2['Fib_1.618']

    diff_day_open_fib_0_382_h2_l1 = day_open - fib_levels_1['Fib_0.382']
    diff_day_open_fib_0_5_h2_l1 = day_open - fib_levels_1['Fib_0.5']
    diff_day_open_fib_0_618_h2_l1 = day_open - fib_levels_1['Fib_0.618']
    diff_day_open_fib_1_5_h2_l1 = day_open - fib_levels_1['Fib_1.5']
    diff_day_open_fib_1_618_h2_l1 = day_open - fib_levels_1['Fib_1.618']

    diff_day_open_fib_0_382_h2_l2 = day_open - fib_levels_2['Fib_0.382']
    diff_day_open_fib_0_5_h2_l2 = day_open - fib_levels_2['Fib_0.5']
    diff_day_open_fib_0_618_h2_l2 = day_open - fib_levels_2['Fib_0.618']
    diff_day_open_fib_1_5_h2_l2 = day_open - fib_levels_2['Fib_1.5']
    diff_day_open_fib_1_618_h2_l2 = day_open - fib_levels_2['Fib_1.618']

    diff_open_fib_current_0_382_h_l = open_val - df.loc[0, 'Fib_current_0.382_H_L']
    diff_open_fib_current_0_5_h_l = open_val - df.loc[0, 'Fib_current_0.5_H_L']
    diff_open_fib_current_0_618_h_l = open_val - df.loc[0, 'Fib_current_0.618_H_L']
    diff_open_fib_current_1_5_h_l = open_val - df.loc[0, 'Fib_current_1.5_H_L']
    diff_open_fib_current_1_618_h_l = open_val - df.loc[0, 'Fib_current_1.618_H_L']

    diff_open_fib_current_0_382_l_h = open_val - df.loc[0, 'Fib_current_0.382_L_H']
    diff_open_fib_current_0_5_l_h = open_val - df.loc[0, 'Fib_current_0.5_L_H']
    diff_open_fib_current_0_618_l_h = open_val - df.loc[0, 'Fib_current_0.618_L_H']
    diff_open_fib_current_1_5_l_h = open_val - df.loc[0, 'Fib_current_1.5_L_H']
    diff_open_fib_current_1_618_l_h = open_val - df.loc[0, 'Fib_current_1.618_L_H']

    volume_difference = volume_1 - volume_2
    volume_percentage_change = ((volume_1 - volume_2) / volume_2) * 100
    volume_ratio = volume_1 / volume_2
    volume_sum = volume_1 + volume_2

    diff_open_day_open = open_val - day_open
    # Create a single row DataFrame for model prediction
    data = {
        'open': open_val,
        'Month': month,
        'Day': day,
        'Year': year,
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
        'Fib_0.382_H1_L1': df.loc[0, 'Fib_0.382_H1_L1'],
        'Fib_0.382_H1_L2': df.loc[0, 'Fib_0.382_H1_L2'],
        'Fib_0.382_H2_L1': df.loc[0, 'Fib_0.382_H2_L1'],
        'Fib_0.382_H2_L2': df.loc[0, 'Fib_0.382_H2_L2'],
        'Fib_0.382_L1_H1': df.loc[0, 'Fib_0.382_L1_H1'],
        'Fib_0.382_L1_H2': df.loc[0, 'Fib_0.382_L1_H2'],
        'Fib_0.382_L2_H1': df.loc[0, 'Fib_0.382_L2_H1'],
        'Fib_0.382_L2_H2': df.loc[0, 'Fib_0.382_L2_H2'],
        'Fib_current_0.382_H_L': df.loc[0, 'Fib_current_0.382_H_L'],
        'Fib_current_0.382_L_H': df.loc[0, 'Fib_current_0.382_L_H'],
        'Fib_0.5_H1_L1': df.loc[0, 'Fib_0.5_H1_L1'],
        'Fib_0.5_H1_L2': df.loc[0, 'Fib_0.5_H1_L2'],
        'Fib_0.5_H2_L1': df.loc[0, 'Fib_0.5_H2_L1'],
        'Fib_0.5_H2_L2': df.loc[0, 'Fib_0.5_H2_L2'],
        'Fib_0.5_L1_H1': df.loc[0, 'Fib_0.5_L1_H1'],
        'Fib_0.5_L1_H2': df.loc[0, 'Fib_0.5_L1_H2'],
        'Fib_0.5_L2_H1': df.loc[0, 'Fib_0.5_L2_H1'],
        'Fib_0.5_L2_H2': df.loc[0, 'Fib_0.5_L2_H2'],
        'Fib_current_0.5_H_L': df.loc[0, 'Fib_current_0.5_H_L'],
        'Fib_current_0.5_L_H': df.loc[0, 'Fib_current_0.5_L_H'],
        'Fib_0.618_H1_L1': df.loc[0, 'Fib_0.618_H1_L1'],
        'Fib_0.618_H1_L2': df.loc[0, 'Fib_0.618_H1_L2'],
        'Fib_0.618_H2_L1': df.loc[0, 'Fib_0.618_H2_L1'],
        'Fib_0.618_H2_L2': df.loc[0, 'Fib_0.618_H2_L2'],
        'Fib_0.618_L1_H1': df.loc[0, 'Fib_0.618_L1_H1'],
        'Fib_0.618_L1_H2': df.loc[0, 'Fib_0.618_L1_H2'],
        'Fib_0.618_L2_H1': df.loc[0, 'Fib_0.618_L2_H1'],
        'Fib_0.618_L2_H2': df.loc[0, 'Fib_0.618_L2_H2'],
        'Fib_current_0.618_H_L': df.loc[0, 'Fib_current_0.618_H_L'],
        'Fib_current_0.618_L_H': df.loc[0, 'Fib_current_0.618_L_H'],
        'Fib_1.5_H1_L1': df.loc[0, 'Fib_1.5_H1_L1'],
        'Fib_1.5_H1_L2': df.loc[0, 'Fib_1.5_H1_L2'],
        'Fib_1.5_H2_L1': df.loc[0, 'Fib_1.5_H2_L1'],
        'Fib_1.5_H2_L2': df.loc[0, 'Fib_1.5_H2_L2'],
        'Fib_1.5_L1_H1': df.loc[0, 'Fib_1.5_L1_H1'],
        'Fib_1.5_L1_H2': df.loc[0, 'Fib_1.5_L1_H2'],
        'Fib_1.5_L2_H1': df.loc[0, 'Fib_1.5_L2_H1'],
        'Fib_1.5_L2_H2': df.loc[0, 'Fib_1.5_L2_H2'],
        'Fib_current_1.5_H_L': df.loc[0, 'Fib_current_1.5_H_L'],
        'Fib_current_1.5_L_H': df.loc[0, 'Fib_current_1.5_L_H'],
        'Fib_1.618_H1_L1': df.loc[0, 'Fib_1.618_H1_L1'],
        'Fib_1.618_H1_L2': df.loc[0, 'Fib_1.618_H1_L2'],
        'Fib_1.618_H2_L1': df.loc[0, 'Fib_1.618_H2_L1'],
        'Fib_1.618_H2_L2': df.loc[0, 'Fib_1.618_H2_L2'],
        'Fib_1.618_L1_H1': df.loc[0, 'Fib_1.618_L1_H1'],
        'Fib_1.618_L1_H2': df.loc[0, 'Fib_1.618_L1_H2'],
        'Fib_1.618_L2_H1': df.loc[0, 'Fib_1.618_L2_H1'],
        'Fib_1.618_L2_H2': df.loc[0, 'Fib_1.618_L2_H2'],
        'Fib_current_1.618_H_L': df.loc[0, 'Fib_current_1.618_H_L'],
        'Fib_current_1.618_L_H': df.loc[0, 'Fib_current_1.618_L_H'],
        'Diff_open_day_open': diff_open_day_open,
        'Diff_open_High_1': diff_open_high_1,
        'Diff_open_Low_1': diff_open_low_1,
        'Diff_open_High_2': diff_open_high_2,
        'Diff_open_Low_2': diff_open_low_2,
        'Diff_open_Fib_0.382_H1_L1': diff_open_fib_0_382_h1_l1,
        'Diff_open_Fib_0.5_H1_L1': diff_open_fib_0_5_h1_l1,
        'Diff_open_Fib_0.618_H1_L1': diff_open_fib_0_618_h1_l1,
        'Diff_open_Fib_1.5_H1_L1': diff_open_fib_1_5_h1_l1,
        'Diff_open_Fib_1.618_H1_L1': diff_open_fib_1_618_h1_l1,
        'Diff_open_Fib_0.382_H1_L2': diff_open_fib_0_382_h1_l2,
        'Diff_open_Fib_0.5_H1_L2': diff_open_fib_0_5_h1_l2,
        'Diff_open_Fib_0.618_H1_L2': diff_open_fib_0_618_h1_l2,
        'Diff_open_Fib_1.5_H1_L2': diff_open_fib_1_5_h1_l2,
        'Diff_open_Fib_1.618_H1_L2': diff_open_fib_1_618_h1_l2,
        'Diff_open_Fib_0.382_H2_L1': diff_open_fib_0_382_h2_l1,
        'Diff_open_Fib_0.5_H2_L1': diff_open_fib_0_5_h2_l1,
        'Diff_open_Fib_0.618_H2_L1': diff_open_fib_0_618_h2_l1,
        'Diff_open_Fib_1.5_H2_L1': diff_open_fib_1_5_h2_l1,
        'Diff_open_Fib_1.618_H2_L1': diff_open_fib_1_618_h2_l1,
        'Diff_open_Fib_0.382_H2_L2': diff_open_fib_0_382_h2_l2,
        'Diff_open_Fib_0.5_H2_L2': diff_open_fib_0_5_h2_l2,
        'Diff_open_Fib_0.618_H2_L2': diff_open_fib_0_618_h2_l2,
        'Diff_open_Fib_1.5_H2_L2': diff_open_fib_1_5_h2_l2,
        'Diff_open_Fib_1.618_H2_L2': diff_open_fib_1_618_h2_l2,
        'Diff_day_open_High_1': diff_day_open_high_1,
        'Diff_day_open_Low_1': diff_day_open_low_1,
        'Diff_day_open_High_2': diff_day_open_high_2,
        'Diff_day_open_Low_2': diff_day_open_low_2,
        'Diff_day_open_Fib_0.382_H1_L1': diff_day_open_fib_0_382_h1_l1,
        'Diff_day_open_Fib_0.5_H1_L1': diff_day_open_fib_0_5_h1_l1,
        'Diff_day_open_Fib_0.618_H1_L1': diff_day_open_fib_0_618_h1_l1,
        'Diff_day_open_Fib_1.5_H1_L1': diff_day_open_fib_1_5_h1_l1,
        'Diff_day_open_Fib_1.618_H1_L1': diff_day_open_fib_1_618_h1_l1,
        'Diff_day_open_Fib_0.382_H1_L2': diff_day_open_fib_0_382_h1_l2,
        'Diff_day_open_Fib_0.5_H1_L2': diff_day_open_fib_0_5_h1_l2,
        'Diff_day_open_Fib_0.618_H1_L2': diff_day_open_fib_0_618_h1_l2,
        'Diff_day_open_Fib_1.5_H1_L2': diff_day_open_fib_1_5_h1_l2,
        'Diff_day_open_Fib_1.618_H1_L2': diff_day_open_fib_1_618_h1_l2,
        'Diff_day_open_Fib_0.382_H2_L1': diff_day_open_fib_0_382_h2_l1,
        'Diff_day_open_Fib_0.5_H2_L1': diff_day_open_fib_0_5_h2_l1,
        'Diff_day_open_Fib_0.618_H2_L1': diff_day_open_fib_0_618_h2_l1,
        'Diff_day_open_Fib_1.5_H2_L1': diff_day_open_fib_1_5_h2_l1,
        'Diff_day_open_Fib_1.618_H2_L1': diff_day_open_fib_1_618_h2_l1,
        'Diff_day_open_Fib_0.382_H2_L2': diff_day_open_fib_0_382_h2_l2,
        'Diff_day_open_Fib_0.5_H2_L2': diff_day_open_fib_0_5_h2_l2,
        'Diff_day_open_Fib_0.618_H2_L2': diff_day_open_fib_0_618_h2_l2,
        'Diff_day_open_Fib_1.5_H2_L2': diff_day_open_fib_1_5_h2_l2,
        'Diff_day_open_Fib_1.618_H2_L2': diff_day_open_fib_1_618_h2_l2,
        'Diff_open_Fib_current_0.382_H_L': diff_open_fib_current_0_382_h_l,
        'Diff_open_Fib_current_0.5_H_L': diff_open_fib_current_0_5_h_l,
        'Diff_open_Fib_current_0.618_H_L': diff_open_fib_current_0_618_h_l,
        'Diff_open_Fib_current_1.5_H_L': diff_open_fib_current_1_5_h_l,
        'Diff_open_Fib_current_1.618_H_L': diff_open_fib_current_1_618_h_l,
        'Diff_open_Fib_current_0.382_L_H': diff_open_fib_current_0_382_l_h,
        'Diff_open_Fib_current_0.5_L_H': diff_open_fib_current_0_5_l_h,
        'Diff_open_Fib_current_0.618_L_H': diff_open_fib_current_0_618_l_h,
        'Diff_open_Fib_current_1.5_L_H': diff_open_fib_current_1_5_l_h,
        'Diff_open_Fib_current_1.618_L_H': diff_open_fib_current_1_618_l_h,
        'Volume_Difference': volume_difference,
        'Volume_Percentage_Change': volume_percentage_change,
        'Volume_Ratio': volume_ratio,
        'Volume_Sum': volume_sum
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
    #prediction_proba = model.predict_proba(input_data)

    st.subheader("Model Prediction")
    st.write("Prediction:", prediction[0])
    #st.write("Prediction Probability:", prediction_proba)

    st.markdown("---")
    st.write("**Note:** Ensure that the retrieved data is correct and there are no missing values before proceeding with model prediction.")

