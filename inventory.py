import streamlit as st
import pandas as pd

def show_inventory():
    st.title("📦 Inventory")

    data = pd.DataFrame({
        "Product": ["Rice", "Oil", "Soap"],
        "Stock": [300, 120, 40]
    })

    st.dataframe(data)
    df = pd.DataFrame({"Sales":[100,200]})
    csv = df.to_csv(index=False)
    st.download_button("Download", csv, "report.csv")