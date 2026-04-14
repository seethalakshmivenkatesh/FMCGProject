import streamlit as st
import numpy as np

def show_dashboard():
    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", "15000", "+5%")
    col2.metric("Demand", "200", "+2%")
    col3.metric("Stock", "700", "-3%")

    st.bar_chart(np.random.randint(100, 300, size=10))