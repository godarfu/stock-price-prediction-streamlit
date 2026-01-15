import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
with open("stock_model.pkl", "rb") as file:
    model = pickle.load(file)


st.title("📈 Stock Price Prediction App")
st.write("Enter stock details to predict closing price")

# User inputs
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=200.0)
low_price = st.number_input("Low Price", value=50.0)
volume = st.number_input("Volume", value=15.0)

if st.button("Predict"):
    # Prediction
    input_data = np.array([[open_price, high_price, low_price, volume]])
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Closing Price: {prediction}")

    # -------- LINE CHART --------
    st.subheader("📈 Stock Price Line Chart")

    x_labels = ["Open", "High", "Low", "Predicted Close"]
    y_values = [open_price, high_price, low_price, prediction]

    fig, ax = plt.subplots()
    ax.plot(x_labels, y_values, marker='o')
    ax.set_xlabel("Price Type")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price Line Visualization")

    st.pyplot(fig)
