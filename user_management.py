import streamlit as st
import json
import os

# Load users
def load_users():
    if os.path.exists("users.json"):
        try:
            with open("users.json", "r") as f:
                return json.load(f)
        except:
            return {"admin": "admin123"}
    return {"admin": "admin123"}

# Save users
def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def manage_users():

    st.title("👤 User Management")

    # only admin allowed
    if st.session_state.get("current_user") != "admin":
        st.error("Access Denied ❌")
        return

    users = load_users()   # ✅ use file

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")

    if st.button("Add User"):

        if new_user in users:
            st.warning("User already exists ❌")

        elif new_user and new_pass:
            users[new_user] = new_pass   # ✅ update dict
            save_users(users)            # ✅ save to file
            st.success("User Added Successfully ✅")

        else:
            st.warning("Enter valid details")