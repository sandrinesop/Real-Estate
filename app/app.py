import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
from src.predict import predict_price
from src.utils import setup_logging


setup_logging(log_filename="app.log")


st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Real Estate Price Predictor")
st.markdown(
    "Enter property details below to get an estimated market price."
)

# Load feature columns to build dynamic form
try:
    feature_cols = joblib.load("models/feature_columns.pkl")
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python main.py` first."
    )
    st.stop()

# ── Input Form ───────────────────────────────────────────────
with st.form("property_form"):
    st.subheader("🏗️ Property Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        year_built = st.number_input(
            "Year Built", min_value=1800, max_value=2025, value=2005
        )
    with col2:
        bedrooms = st.number_input(
            "Bedrooms", min_value=1, max_value=20, value=3
        )
    with col3:
        bathrooms = st.number_input(
            "Bathrooms", min_value=1, max_value=20, value=2
        )

    col4, col5 = st.columns(2)
    with col4:
        sqft = st.number_input(
            "Square Footage", min_value=100, max_value=20000, value=1500
        )
    with col5:
        property_type = st.selectbox(
            "Property Type",
            options=["House", "Condo", "Townhouse"]
        )

    model_choice = st.selectbox(
        "Prediction Model",
        options=["random_forest", "linear_regression"],
        format_func=lambda x: (
            "Random Forest (Recommended)" if x == "random_forest"
            else "Linear Regression"
        )
    )

    submitted = st.form_submit_button("💰 Predict Price")

# ── Prediction ───────────────────────────────────────────────
if submitted:
    try:
        # Build input dictionary
        input_dict = {col: 0 for col in feature_cols}

        # Fill in known values
        if 'year_built' in input_dict:
            input_dict['year_built'] = year_built
        if 'bedrooms' in input_dict:
            input_dict['bedrooms'] = bedrooms
        if 'bathrooms' in input_dict:
            input_dict['bathrooms'] = bathrooms
        if 'sqft' in input_dict:
            input_dict['sqft'] = sqft

        # Set property type one-hot
        if property_type == "Condo" and 'property_type_Condo' in input_dict:
            input_dict['property_type_Condo'] = 1
        elif property_type == "Townhouse" and 'property_type_Townhouse' in input_dict:
            input_dict['property_type_Townhouse'] = 1

        result = predict_price(input_dict, model_name=model_choice)
        price = result['predicted_price']

        st.divider()
        st.subheader("Estimated Property Price")
        st.success(f"## 💰 ${price:,.0f}")
        st.caption(f"Predicted by: `{model_choice}`")

        # Show input summary
        st.subheader("📋 Property Summary")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Year Built", year_built)
        col_b.metric("Bedrooms", bedrooms)
        col_c.metric("Bathrooms", bathrooms)
        col_d.metric("Sq. Footage", f"{sqft:,}")

    except FileNotFoundError as e:
        st.error(f"Model not found. Run `python main.py` first.\n\n{e}")
    except ValueError as e:
        st.warning(f"⚠️ Invalid input: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

st.divider()
st.caption(
    "Models: Linear Regression & Random Forest | "
    "Success Criteria: MAE < $70,000"
)