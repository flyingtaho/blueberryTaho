# app.py
import streamlit as st
from solver import explore_ranges, find_optimal_input, send_to_api

st.set_page_config(page_title="MLE Challenge Solver", layout="centered")
st.title("🚀 MLE Challenge: Probability ≥ 0.999 Finder")

st.markdown("""
This app automatically explores your training data ranges,
finds valid input values that yield a logistic regression probability ≥ 0.999,
and submits them to the API endpoint to retrieve the password.
""")

# Upload datasets
app_file = st.file_uploader("Upload dummy_application_data.csv", type="csv")
travel_file = st.file_uploader("Upload dummy_travel_data.csv", type="csv")

if app_file and travel_file:
    with open("dummy_application_data.csv", "wb") as f:
        f.write(app_file.read())
    with open("dummy_travel_data.csv", "wb") as f:
        f.write(travel_file.read())

    st.write("📊 Exploring data ranges...")
    ranges = explore_ranges("dummy_application_data.csv", "dummy_travel_data.csv")
    st.json(ranges)

    st.write("🔍 Searching for valid sample data...")
    data, prob = find_optimal_input(ranges)

    if data:
        st.success(f"Found valid input with probability = {prob:.6f}")
        st.json(data)

        if st.button("Send to API"):
            response = send_to_api(data)
            st.info(f"Server response: {response}")
    else:
        st.error("No valid input found. Try checking dataset ranges.")
