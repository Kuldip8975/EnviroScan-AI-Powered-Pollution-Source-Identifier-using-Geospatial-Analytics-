import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="🌍 AI-EnviroScan Dashboard",
    layout="wide",
    page_icon="🌿"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/pollution_data_comprehensive.csv")
    return df

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("models/pollution_source_model_artifacts.joblib")
    return model

# Main Dashboard Function
def main():
    st.title("🌍 AI-EnviroScan Dashboard")
    st.markdown("### Real-Time Pollution Monitoring & Source Prediction System")

    df = load_data()
    model = load_model()

    # Sidebar filters
    st.sidebar.header("🔎 Filters")
    city = st.sidebar.selectbox("Select City", df["City"].unique())
    pollutant = st.sidebar.selectbox("Select Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])

    filtered_df = df[df["City"] == city]

    # Display stats
    st.subheader(f"📈 Pollution Data for {city}")
    st.write(filtered_df.describe())

    # Visualization
    fig = px.line(filtered_df, x="Date", y=pollutant, title=f"{pollutant} Levels in {city} Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Prediction section
    st.subheader("🤖 Predict Pollution Source")
    pm25 = st.number_input("PM2.5", 0.0, 500.0, 50.0)
    pm10 = st.number_input("PM10", 0.0, 500.0, 80.0)
    no2 = st.number_input("NO2", 0.0, 500.0, 30.0)
    so2 = st.number_input("SO2", 0.0, 500.0, 15.0)
    co = st.number_input("CO", 0.0, 10.0, 0.8)
    o3 = st.number_input("O3", 0.0, 500.0, 25.0)

    if st.button("Predict Source"):
        features = np.array([[pm25, pm10, no2, so2, co, o3]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Source: **{prediction}**")

# Run app
if __name__ == "__main__":
    main()

}
