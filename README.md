# 📈 Demand Forecasting & Inventory Optimization

Production-ready AI solution for FMCG distribution using **Prophet + LightGBM Hybrid Model**.

## 🚀 Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the app**
```bash
streamlit run app.py
```

3. **Open browser** → `http://localhost:8501`

## 📁 Features

- **Professional Streamlit Dashboard** with sidebar navigation
- **Prophet**: Time series forecasting with seasonality
- **LightGBM**: ML model with lag/rolling features  
- **Hybrid Ensemble**: Weighted combination of both models
- **Weather Integration**: External demand factors
- **Inventory Optimization**: Smart reorder recommendations
- **Production-quality code** with error handling

## 🗂️ Expected CSV Format

```csv
date,sales,product_id,location
2024-01-01,150,PROD001,Coimbatore
2024-01-02,165,PROD001,Coimbatore
...
```

## 🔧 Folder Structure
