import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Load dataset
file_path = "/Users/gehnayadav/Desktop/stock_presiction/stock_data_kaggle.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

# Streamlit UI
st.title("📈 Big Tech Stock Predictor (2010–2025)")
st.write("Historical and Predictive Analysis of AAPL, AMZN, MSFT, GOOGL, NVDA")

stocks = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'NVDA']
stock_choice = st.selectbox("Choose a stock to predict", stocks)

close_col = f'Close_{stock_choice}'
st.subheader(f"Raw Closing Price of {stock_choice}")
st.line_chart(df[['Date', close_col]].set_index('Date'))

# Prepare data
data = df[['Date', close_col]].dropna().copy()
data.set_index('Date', inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create time series sequences
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
dates = data.index[60:]  # aligns with y

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
dates_test = dates[len(X_train):]

# Reshape
X_train_lr = X_train.reshape(X_train.shape[0], -1)
X_test_lr = X_test.reshape(X_test.shape[0], -1)

# Train using the sklearn library
model = LinearRegression()
model.fit(X_train_lr, y_train)

# Predict
y_pred = model.predict(X_test_lr)

# Inverse transform
y_test_rescaled = scaler.inverse_transform(y_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Plot
st.subheader(f"{stock_choice} Stock Price: Actual vs Predicted")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates_test, y_test_rescaled, label='Actual Price')
ax.plot(dates_test, y_pred_rescaled, label='Predicted Price')
ax.set_title(f"{stock_choice} Price Prediction")
ax.set_ylabel("USD")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig)