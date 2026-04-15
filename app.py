"""
Streamlit Dashboard for Demand Forecasting & Inventory Optimization
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from datetime import datetime, timedelta

# Import project modules
from utils import load_data, preprocess_data, calculate_inventory_metrics, create_sample_data
from model import DemandForecaster
from api import WeatherAPI

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'feature_info' not in st.session_state:
    st.session_state.feature_info = None
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None
# Page configuration
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class ForecastingApp:
    def __init__(self):
        self.df = st.session_state.df
        self.processed_df = st.session_state.processed_df
        self.forecaster = st.session_state.forecaster
        self.weather_api = WeatherAPI()
        self.feature_info = st.session_state.feature_info
    def sidebar_navigation(self):
        """Create professional sidebar navigation"""
        st.sidebar.title("🚀 Navigation")
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Select Page",
            ["🏠 Home", "📁 Upload Data", "📊 View Data", "🤖 Train Model", 
             "📈 Forecast", "📦 Inventory"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("**Built for FMCG Distribution**")
        st.sidebar.caption("Prophet + LightGBM Hybrid Model")
        
        return page
    def home_page(self):
        """Home page with KPIs and project info"""
        st.markdown('<h1 class="main-header">📊 Demand Forecasting & Inventory Optimization</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### AI-Powered FMCG Distribution Solution
        - **Prophet**: Captures seasonal trends and holidays
        - **LightGBM**: Leverages engineered features and lags  
        - **Hybrid Model**: Combines both for optimal accuracy
        - **Weather Integration**: External factors affecting demand
        - **Inventory Optimization**: Smart reorder recommendations
        """)
        
        # Create sample KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", "₹4.2L", "12%")
        with col2:
            st.metric("Avg Daily Demand", "165 units", "3%")
        with col3:
            st.metric("Forecast Accuracy", "92.5%", "2%")
        with col4:
            st.metric("Stockouts", "2 days", "-50%")
        
        st.markdown("---")
        st.info("👆 Upload your sales data (CSV) to get started!")
    def upload_data_page(self):
        """Upload and store CSV data"""
        st.header("📁 Upload Sales Data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose CSV file (date, sales columns required)",
            type="csv",
            help="Expected format: date,sales,product_id,location"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            save_path = "data/uploaded_sales.csv"
            os.makedirs("data", exist_ok=True)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and display preview
            @st.cache_data
            def load_uploaded():
                return load_data(save_path)
            
            self.df = load_uploaded()
            st.session_state.df = self.df
            st.success(f"✅ Data uploaded! Shape: {self.df.shape}")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(self.df.head(10), use_container_width=True)
            
        elif st.button("🧪 Use Sample Data"):
            self.df = create_sample_data()
            st.session_state.df = self.df
            st.success("✅ Sample data loaded!")