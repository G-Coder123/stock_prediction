# 📈 Big Tech Stock Predictor (2010–2025)

This is a **Streamlit web app** built in Python that visualizes and predicts stock prices for five leading tech companies: **Apple (AAPL), Amazon (AMZN), Microsoft (MSFT), Google (GOOGL), and NVIDIA (NVDA)**, using 15 years of historical data.

---

## 📂 Dataset Overview

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/marianadeem755/stock-market-data)
- **Duration**: January 1, 2010 – January 1, 2025
- **Data Fields**: 
  - Open, High, Low, Close prices
  - Volume
  - Separate columns per stock (e.g., `Close_AAPL`, `Open_AMZN`)

---

## 🔍 Features

- 📊 Interactive visualizations of raw stock prices
- 🤖 Predictive model using **Linear Regression**
- ⏳ Uses **MinMaxScaler** for normalization and sliding windows of historical data
- 🖼️ Streamlit + Matplotlib for dynamic and readable visual outputs

---

## 📌 Requirements

Ensure the following Python packages are installed:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
