# Stock Market Prediction using LightGBM on NIFTY Data

## Overview

This project focuses on the development of a robust model to predict stock market movements for the NIFTY index using historical and minute-level trading data. The model is built using LightGBM, a highly efficient gradient boosting framework, and the predictions are categorized into "buy," "sell," or "wait" signals. This repository provides both the code for model training and a Streamlit web application that allows users to interact with the model and make predictions on real-time data.

## Project Features

- **Data Integration:** The project integrates daily NIFTY data from Yahoo Finance with minute-level trading data to create a comprehensive dataset for model training.
- **Feature Engineering:** Key technical indicators, including Fibonacci levels and volume differences, are computed and included in the dataset to enhance the predictive power of the model.
- **Model Training:** The LightGBM model is trained using a stratified K-Fold cross-validation approach, optimizing for multi-class log loss. The final model achieves high accuracy in predicting stock price movements.
- **Feature Importance Analysis:** The project includes detailed analysis of feature importance to identify the most influential factors in the model's predictions.
- **Streamlit Web Application:** A user-friendly Streamlit app is developed to allow users to input current market conditions and get real-time predictions from the trained model.

## Streamlit Application

### Link to the App
[Stock Prediction Model on Streamlit](https://sathyanifty50.streamlit.app/)

### Key Features of the App:
- **Interactive Input:** Users can input current market conditions such as open prices, high/low values, and volumes through an intuitive sidebar.
- **Real-Time Data Retrieval:** The app fetches the latest trading data for the NIFTY index from Yahoo Finance.
- **Fibonacci Levels Calculation:** The app dynamically calculates Fibonacci retracement levels based on recent high and low values.
- **Prediction Results:** The model provides predictions for the next market move, along with the probability of each outcome, helping users make informed decisions.

### How to Use the App:
1. **Input Parameters:** Use the sidebar to input the relevant stock data. The app allows you to manually input values for the open price, high, low, and volume, or you can retrieve the latest data from Yahoo Finance.
2. **Select Date and Time:** Specify the date and time for the market conditions you are interested in predicting.
3. **View Aggregated Data:** The app will display the aggregated data that will be used for prediction, including calculated technical indicators.
4. **Model Prediction:** The model will predict whether the stock is a "buy," "sell," or "wait," along with the prediction probabilities.

## Installation and Setup

To run this project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Sathyavrv/Nifty.git
cd Nifty
```

### 2. Install the Required Dependencies
Make sure you have Python 3.10+ installed. Install the required libraries using:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App Locally
```bash
streamlit run app.py
```

This will start a local server and open the app in your default web browser.

### 4. Accessing the Model
The pre-trained LightGBM model is included in the repository as lgb_model_june11.pkl. The app loads this model to make predictions based on the user inputs.

## About the Project
This project demonstrates advanced data processing and machine learning techniques, showcasing my expertise in financial data modeling. The integration of real-time data, advanced feature engineering, and a robust prediction model into a user-friendly web application exemplifies the ability to create practical solutions with real-world impact.

The codebase is well-documented and structured, ensuring that any employer or collaborator who reviews this repository can quickly understand the methodologies used and the capabilities demonstrated.

## Contribution
Contributions to this project are welcome! Feel free to fork the repository, make enhancements, or suggest new features. Please open an issue to discuss any significant changes.
