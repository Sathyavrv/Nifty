import streamlit as st
import pandas as pd
import yfinance as yf
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

# Function to calculate Fibonacci levels
def calculate_fibonacci_levels(df):
    fib_ratios = [0.382, 0.5, 0.618, 0.786, 1.5, 1.618]
    high_low_combinations = [
        ('High_1', 'Low_1'), ('High_1', 'Low_2'), ('High_2', 'Low_1'), ('High_2', 'Low_2'),
        ('Low_1', 'High_1'), ('Low_1', 'High_2'), ('Low_2', 'High_1'), ('Low_2', 'High_2'),
        ('Current_High', 'Current_Low'), ('Current_Low', 'Current_High')
    ]

    for high, low in high_low_combinations:
        for ratio in fib_ratios:
            df[f'Fib_{ratio}_{high}_{low}'] = df[high] - (df[high] - df[low]) * ratio

    return df

# Function to calculate differences between all specified columns
def calculate_all_differences(df, base_columns, diff_columns):
    new_cols = {}

    for base_col in base_columns:
        for col in diff_columns:
            if base_col != col:
                new_cols[f'Diff_{base_col}_{col}'] = df[base_col] - df[col]

    # Concatenate all new columns at once to avoid DataFrame fragmentation
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    return df

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

    # Apply the functions
    df = calculate_fibonacci_levels(df)
    column_pairs = [('Current_High', 'High_1'), ('Current_High', 'High_2'), ('Current_Low', 'Low_1'),
                    ('Current_Low', 'Low_2'), ('High_1', 'High_2'), ('Low_1', 'Low_2')]
    for col1, col2 in column_pairs:
        df[f'Diff_{col1}_{col2}'] = df[col1] - df[col2]

    # Ensure all features are included in the correct order
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
    }

    # Adding calculated differences and Fibonacci levels in the correct order
    features_in_order = [
        'Diff_Current_High_High_1', 'Diff_Current_High_High_2', 'Diff_Current_Low_Low_1',
        'Diff_Current_Low_Low_2', 'Diff_High_1_High_2', 'Diff_Low_1_Low_2'
    ]

    for high, low in [('High_1', 'Low_1'), ('High_1', 'Low_2'), ('High_2', 'Low_1'), ('High_2', 'Low_2'),
                      ('Low_1', 'High_1'), ('Low_1', 'High_2'), ('Low_2', 'High_1'), ('Low_2', 'High_2'),
                      ('Current_High', 'Current_Low'), ('Current_Low', 'Current_High')]:
        for ratio in fib_ratios:
            features_in_order.append(f'Fib_{ratio}_{high}_{low}')

    for col1, col2 in column_pairs:
        features_in_order.append(f'Diff_{col1}_{col2}')

    # Additional features based on 'open' and 'day_open'
    additional_features_ordered = [
        'Diff_open_day_open', 'Diff_open_Current_High', 'Diff_open_Current_Low', 'Diff_open_High_1',
        'Diff_open_Low_1', 'Diff_open_Volume_1', 'Diff_open_High_2', 'Diff_open_Low_2', 'Diff_open_Volume_2',
        'Diff_open_Diff_Current_High_High_1', 'Diff_open_Diff_Current_High_High_2',
        'Diff_open_Diff_Current_Low_Low_1', 'Diff_open_Diff_Current_Low_Low_2',
        'Diff_open_Diff_High_1_High_2', 'Diff_open_Diff_Low_1_Low_2',
        'Volume_Difference', 'Volume_Percentage_Change', 'Volume_Ratio', 'Volume_Sum',
        'High_Low_Difference', 'VWAP', '3D_Volume_MA', '5D_Volume_MA', '7D_Volume_MA', '1D_Volume_MA',
        '2D_Volume_MA'
    ]

    # Combining all features
    all_features = list(data.keys()) + features_in_order + additional_features_ordered

    # Update the 'data' dictionary with all features
    for feature in all_features:
        if feature not in data:
            data[feature] = df[feature].iloc[0] if feature in df.columns else 0

    # Convert data to DataFrame for prediction
    data_df = pd.DataFrame([data])

    # Display the aggregated row
    st.subheader("Aggregated Data for Model Prediction")
    st.dataframe(data_df)

    # Show the number of features
    st.write(f"Total number of features: {len(data_df.columns)}")

    # Load the model
    model = load_model()

    prediction = model.predict(data_df)
    prediction_proba = model.predict_proba(data_df)

    st.subheader("Model Prediction")
    st.write("Prediction:", prediction[0])
    st.write("Prediction Probability:", prediction_proba)

    st.markdown("---")
    st.write("**Note:** Ensure that the retrieved data is correct and there are no missing values before proceeding with model prediction.")
