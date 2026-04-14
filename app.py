import streamlit as st
from login import login
from dashboard import show_dashboard
from inventory import show_inventory
from user_management import manage_users

st.set_page_config(layout="wide")

if "login" not in st.session_state:
    st.session_state["login"] = False

# LOGIN
if not st.session_state["login"]:
    login()

# AFTER LOGIN
else:
    st.sidebar.title("Menu")

    st.sidebar.markdown(f"👤 **{st.session_state.get('current_user', '')}**")

    menu = st.sidebar.radio("Menu", ["Dashboard", "Inventory", "User Management"])
    if st.sidebar.button("⏻ Logout"):
        st.session_state["login"] = False
        st.session_state["current_user"] = ""
        st.rerun()
    elif menu == "Dashboard":
        show_dashboard()

    elif menu == "Inventory":
        show_inventory()
    elif menu == "User Management":
        manage_users()