import streamlit as st
import json
import os

def load_users():
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except:
        return {"admin": "admin123"}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def login():
    st.markdown("<h2 style='text-align: center;'>🏪 Demand Forecasting & Inventory Optimization</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>for FMCG Distribution Using AI Models</h4>", unsafe_allow_html=True)

    st.title("🔐 Login")

    users = load_users()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in users and users[username] == password:
            st.session_state["login"] = True
            st.session_state["current_user"] = username
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")