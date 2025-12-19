# streamlit_dashboard_final.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI-EnviroScan India - Real-Time Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# GLOBAL CSS LOADER
# ----------------------------------------------------------
def load_custom_css():
    custom_css = """
    <style>

    body {
        background-color: #f4f5f7 !important;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }

    /* INDIA THEME HEADER (LIKE YOUR IMAGE) */
    .india-header {
        background: linear-gradient(90deg, #FFA756, #FFFFFF, #4CAF50);
        border-radius: 18px;
        padding: 28px 30px;
        border: 2px solid #002060;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.12);
    }

    .india-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #002060;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .india-subtitle {
        font-size: 1.2rem;
        color: #002060;
        margin-bottom: 6px;
    }

    .india-tagline {
        font-size: 0.95rem;
        color: #003399;
        opacity: 0.9;
    }

    /* SECTION TITLES */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2933;
        padding-bottom: 6px;
        border-bottom: 1px solid #e1e7ee;
        margin-bottom: 12px;
    }

    /* KPI CARDS */
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e1e7ee;
        padding: 10px 12px;
        box-shadow: 0 2px 6px rgba(15,23,42,0.04);
        margin-bottom: 10px;
    }

    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #7b8794;
        letter-spacing: 0.05em;
    }

    .kpi-value {
        font-size: 1.35rem;
        font-weight: 600;
        color: #102a43;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #8292a0;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e1e7ee !important;
    }

    /* TABS */
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    .footer-text {
        font-size: 0.8rem;
        color: #8292a0;
        text-align: center;
    }

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# ----------------------------------------------------------
# MAIN DASHBOARD CLASS
# ----------------------------------------------------------
class FinalEnviroScanDashboard:

    def __init__(self):
        self.POLLUTION_THRESHOLDS = {
            "PM2.5": {"good": 12, "moderate": 35, "unhealthy": 55, "hazardous": 150},
            "PM10":  {"good": 54, "moderate": 154, "unhealthy": 254, "hazardous": 424},
            "NO2":   {"good": 40, "moderate": 100, "unhealthy": 360, "hazardous": 649}
        }
        self.df = None

    # ------------------------ EMAILJS ----------------------
    @staticmethod
    def send_emailjs_alert(to_email, subject, message):
        SERVICE_ID = "service_r4tcqtk"
        TEMPLATE_ID = "template_obngge7"
        PUBLIC_KEY = "5dwOue8RZ31gNgO_y"

        url = "https://api.emailjs.com/api/v1.0/email/send"

        data = {
            "service_id": SERVICE_ID,
            "template_id": TEMPLATE_ID,
            "user_id": PUBLIC_KEY,
            "template_params": {
                "to_email": to_email,
                "subject": subject,
                "message": message
            }
        }

        try:
            response = requests.post(url, json=data)
            print("EmailJS status:", response.status_code, response.text)
        except Exception as e:
            print("EmailJS error:", e)

    # ------------------------ DATA LOAD --------------------
    def load_data(self):
        with st.spinner("🔄 Loading All India pollution data..."):
            try:
                self.df = pd.read_csv("data/pollution_data_all_india_enhanced.csv")
                st.success("✅ Enhanced All India dataset loaded.")
            except Exception:
                try:
                    self.df = pd.read_csv("data/pollution_data_all_india.csv")
                    st.warning("⚠ Using basic All India dataset.")
                except Exception:
                    st.warning("⚠ No data file found. Generating sample data...")
                    self.create_sample_data()

        self.df["timestamp"] = pd.to_datetime(self.df.get("timestamp", datetime.now()), errors="coerce")
        self.df.dropna(subset=["timestamp"], inplace=True)

        if "zone" not in self.df.columns:
            self.df["zone"] = "Unknown"
        if "sensor_latitude" not in self.df.columns:
            self.df["sensor_latitude"] = 20 + np.random.uniform(-5, 5, len(self.df))
        if "sensor_longitude" not in self.df.columns:
            self.df["sensor_longitude"] = 78 + np.random.uniform(-5, 5, len(self.df))

    # ------------------------ SAMPLE DATA ------------------
    def create_sample_data(self):
        cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Hyderabad"]
        zones = ["North", "West", "East", "South", "South", "South"]

        dates = pd.date_range(end=datetime.now(), periods=7 * 24, freq="H")
        rows = []

        for city, zone in zip(cities, zones):
            for s in range(1, 3):
                for t in dates:
                    hour = t.hour
                    rush_factor = 1.3 if (7 <= hour <= 10 or 17 <= hour <= 21) else 1.0
                    base_pm25 = np.random.uniform(40, 90)
                    base_pm10 = np.random.uniform(80, 180)

                    rows.append({
                        "timestamp": t,
                        "city": city,
                        "zone": zone,
                        "sensor_name": f"{city}_Sensor_{s}",
                        "sensor_latitude": 20 + np.random.uniform(-5, 5),
                        "sensor_longitude": 78 + np.random.uniform(-5, 5),
                        "PM2.5": max(10, base_pm25 * rush_factor * np.random.uniform(0.8, 1.2)),
                        "PM10": max(20, base_pm10 * rush_factor * np.random.uniform(0.8, 1.2)),
                        "NO2": np.random.uniform(20, 100),
                        "SO2": np.random.uniform(5, 40),
                        "CO": np.random.uniform(0.5, 3.0),
                        "O3": np.random.uniform(10, 70),
                        "temperature_c": np.random.uniform(18, 40),
                        "humidity": np.random.uniform(35, 90),
                        "wind_speed": np.random.uniform(2, 30),
                        "area_type": np.random.choice(["Industrial", "Residential", "Commercial", "Traffic"])
                    })

        self.df = pd.DataFrame(rows)
        st.success("✅ Sample data generated with realistic patterns.")

    # ------------------ SIMPLE SOURCE MODEL ----------------
    def predict_pollution_source(self, pollution_data):
        sources = ["Industrial", "Vehicular", "Construction", "Agricultural", "Natural"]

        pm25 = pollution_data.get("PM2.5", 50)
        pm10 = pollution_data.get("PM10", 100)
        no2 = pollution_data.get("NO2", 30)

        probs = np.array([
            0.3 * (pm25 / 100) + 0.2 * (pm10 / 200),
            0.4 * (no2 / 50) + 0.1 * (pm25 / 100),
            0.6 * (pm10 / 250),
            0.2,
            0.1 * (50 / pm25) if pm25 > 0 else 0.1
        ])

        probs = np.clip(probs, 0.001, None)
        probs = probs / probs.sum()
        return sources, probs

    # ----------------------- ALERT CHECK -------------------
    def check_pollution_alerts(self, pollution_data):
        alerts = []
        for pollutant, value in pollution_data.items():
            if pollutant not in self.POLLUTION_THRESHOLDS:
                continue
            th = self.POLLUTION_THRESHOLDS[pollutant]
            if value >= th["hazardous"]:
                alerts.append({"level": "CRITICAL", "message": f"{pollutant} at {value:.1f} μg/m³ — HAZARDOUS!"})
            elif value >= th["unhealthy"]:
                alerts.append({"level": "WARNING", "message": f"{pollutant} at {value:.1f} μg/m³ — Unhealthy."})
            elif value >= th["moderate"]:
                alerts.append({"level": "NOTICE", "message": f"{pollutant} at {value:.1f} μg/m³ — Moderate."})
        return alerts

    # --------------------- SIDEBAR INPUTS ------------------
    def create_sidebar_inputs(self):
        st.sidebar.header("🎯 Dashboard Controls")

        zones = sorted(self.df["zone"].dropna().unique())
        selected_zone = st.sidebar.selectbox("🗺️ Zone / State", ["All Zones"] + zones)

        if selected_zone == "All Zones":
            cities = sorted(self.df["city"].dropna().unique())
        else:
            cities = sorted(self.df.loc[self.df["zone"] == selected_zone, "city"].dropna().unique())

        selected_city = st.sidebar.selectbox("🏙️ City", cities)

        input_method = st.sidebar.radio(
            "📍 Data Mode",
            ["Real-time Data", "Manual Input", "Historical Analysis"]
        )

        pollution_data = {}
        lat, lon = 20.5937, 78.9629

        if input_method == "Manual Input":
            st.sidebar.subheader("🧪 Manual Pollutant Levels")
            pollution_data = {
                "PM2.5": st.sidebar.slider("PM2.5 (μg/m³)", 0, 500, 80),
                "PM10":  st.sidebar.slider("PM10 (μg/m³)",  0, 600, 150),
                "NO2":   st.sidebar.slider("NO2 (μg/m³)",   0, 200, 40),
                "SO2":   st.sidebar.slider("SO2 (μg/m³)",   0, 100, 15),
                "CO":    st.sidebar.slider("CO (mg/m³)",    0.0, 10.0, 1.2),
                "O3":    st.sidebar.slider("O3 (μg/m³)",    0, 200, 45),
            }
            st.sidebar.subheader("🌤️ Environment")
            pollution_data.update({
                "temperature_c": st.sidebar.slider("Temperature (°C)", -10, 50, 28),
                "humidity":      st.sidebar.slider("Humidity (%)", 0, 100, 60),
                "wind_speed":    st.sidebar.slider("Wind Speed (km/h)", 0, 100, 12),
            })
            row = self.df[self.df["city"] == selected_city].iloc[0]
            lat, lon = row["sensor_latitude"], row["sensor_longitude"]

        elif input_method == "Historical Analysis":
            st.sidebar.subheader("📅 Historical Range")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=7))
            with col2:
                end_date = st.date_input("End date", value=datetime.now().date())

            city_data = self.df[
                (self.df["city"] == selected_city) &
                (self.df["timestamp"].dt.date >= start_date) &
                (self.df["timestamp"].dt.date <= end_date)
            ]

            if not city_data.empty:
                pollution_data = {
                    "PM2.5": city_data["PM2.5"].mean(),
                    "PM10":  city_data["PM10"].mean(),
                    "NO2":   city_data["NO2"].mean(),
                    "SO2":   city_data.get("SO2", pd.Series([0]*len(city_data))).mean(),
                    "CO":    city_data.get("CO", pd.Series([0]*len(city_data))).mean(),
                    "O3":    city_data.get("O3", pd.Series([0]*len(city_data))).mean(),
                    "temperature_c": city_data.get("temperature_c", pd.Series([0]*len(city_data))).mean(),
                    "humidity":      city_data.get("humidity", pd.Series([0]*len(city_data))).mean(),
                    "wind_speed":    city_data.get("wind_speed", pd.Series([0]*len(city_data))).mean()
                }
            else:
                pollution_data = {k: 0 for k in ["PM2.5","PM10","NO2","SO2","CO","O3","temperature_c","humidity","wind_speed"]}

            row = self.df[self.df["city"] == selected_city].iloc[0]
            lat, lon = row["sensor_latitude"], row["sensor_longitude"]

        else:
            st.sidebar.caption("Using last 1-hour sensor data for this city.")
            latest_time = self.df["timestamp"].max()
            cutoff = latest_time - timedelta(hours=1)
            city_data = self.df[
                (self.df["city"] == selected_city) &
                (self.df["timestamp"] >= cutoff)
            ]
            if not city_data.empty:
                pollution_data = {
                    "PM2.5": city_data["PM2.5"].mean(),
                    "PM10":  city_data["PM10"].mean(),
                    "NO2":   city_data["NO2"].mean(),
                    "SO2":   city_data.get("SO2", pd.Series([0]*len(city_data))).mean(),
                    "CO":    city_data.get("CO", pd.Series([0]*len(city_data))).mean(),
                    "O3":    city_data.get("O3", pd.Series([0]*len(city_data))).mean(),
                    "temperature_c": city_data.get("temperature_c", pd.Series([0]*len(city_data))).mean(),
                    "humidity":      city_data.get("humidity", pd.Series([0]*len(city_data))).mean(),
                    "wind_speed":    city_data.get("wind_speed", pd.Series([0]*len(city_data))).mean()
                }
            else:
                pollution_data = {k: 0 for k in ["PM2.5","PM10","NO2","SO2","CO","O3","temperature_c","humidity","wind_speed"]}
            row = self.df[self.df["city"] == selected_city].iloc[0]
            lat, lon = row["sensor_latitude"], row["sensor_longitude"]

        return selected_city, lat, lon, pollution_data, input_method

    # ------------------ AI ANALYSIS TAB --------------------
    def create_prediction_section(self, city, pollution_data):
        st.markdown('<div class="section-title">AI-Based Source Analysis</div>', unsafe_allow_html=True)

        sources, probs = self.predict_pollution_source(pollution_data)

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">City</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{city}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-sub">Selected location</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        for i, key in enumerate(["PM2.5", "PM10", "NO2"], start=1):
            with kpi_cols[i]:
                value = pollution_data.get(key, 0)
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-label">{key}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-value">{value:.1f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="kpi-sub">μg/m³</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        left, right = st.columns([1.4, 1])

        critical_pollutants = []
        with left:
            st.write("**Source Confidence Breakdown**")
            for src, p in zip(sources, probs):
                confidence = p * 100
                if confidence > 70:
                    color = "#d64545"
                elif confidence > 50:
                    color = "#f0b429"
                else:
                    color = "#3e862e"
                st.markdown(
                    f"""
                    <div style="margin:6px 0;padding:8px 10px;background:#ffffff;border-radius:8px;
                    border:1px solid #e1e7ee;">
                        <div style="display:flex;justify-content:space-between;">
                            <span style="font-size:0.9rem;font-weight:500;color:#102a43;">{src}</span>
                            <span style="font-size:0.9rem;color:#52606d;">{confidence:.1f}%</span>
                        </div>
                        <div style="background:#edf1f7;border-radius:4px;height:6px;margin-top:6px;">
                            <div style="background:{color};width:{confidence}%;height:6px;border-radius:4px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with right:
            st.write("**Source Contribution**")
            fig = px.pie(values=probs, names=sources, hole=0.45)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=320, margin=dict(t=10, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title" style="margin-top:12px;">Current Air Quality Status</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, pol in enumerate(["PM2.5", "PM10", "NO2"]):
            value = pollution_data.get(pol, 0)
            with cols[i]:
                th = self.POLLUTION_THRESHOLDS[pol]
                if value >= th["hazardous"]:
                    st.error(f"**{pol}**\n\n{value:.1f} μg/m³\n\n🔴 HAZARDOUS")
                    critical_pollutants.append(pol)
                elif value >= th["unhealthy"]:
                    st.warning(f"**{pol}**\n\n{value:.1f} μg/m³\n\n🟠 Unhealthy")
                    critical_pollutants.append(pol)
                elif value >= th["moderate"]:
                    st.info(f"**{pol}**\n\n{value:.1f} μg/m³\n\n🟡 Moderate")
                else:
                    st.success(f"**{pol}**\n\n{value:.1f} μg/m³\n\n✅ Good")

        return critical_pollutants

    # ------------------ TRENDS TAB -------------------------
    def create_trends_section(self, selected_city):
        st.markdown('<div class="section-title">Pollution Trends Over Time</div>', unsafe_allow_html=True)

        if self.df is None:
            st.error("❌ Data not loaded.")
            return

        if selected_city not in self.df["city"].unique():
            st.warning(f"⚠ No data for {selected_city}.")
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            days = st.slider("Time window (days)", 1, 60, 7)
        with col2:
            pollutant = st.selectbox("Pollutant", ["PM2.5", "PM10", "NO2"])

        df_city = self.df[self.df["city"] == selected_city].copy()
        df_city["timestamp"] = pd.to_datetime(df_city["timestamp"], errors="coerce")
        df_city.dropna(subset=["timestamp"], inplace=True)

        if df_city.empty:
            st.warning("No data available for this city.")
            return

        max_date = df_city["timestamp"].max()
        min_date = max_date - timedelta(days=days)
        df_filtered = df_city[(df_city["timestamp"] >= min_date) & (df_city["timestamp"] <= max_date)]

        if df_filtered.empty:
            st.warning(f"No trend data for last {days} days.")
            return

        fig = px.line(df_filtered, x="timestamp", y=pollutant, markers=True)
        fig.update_traces(line_shape="spline")
        fig.update_layout(
            xaxis_title="Date & Time",
            yaxis_title=f"{pollutant} (μg/m³)",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ MAP TAB ----------------------------
    def create_map_section(self, city, lat, lon):
        st.markdown('<div class="section-title">Geospatial Air Quality Distribution</div>', unsafe_allow_html=True)

        city_data = self.df[self.df["city"] == city]
        if city_data.empty:
            st.info("No location data for this city.")
            return

        fig = px.scatter_mapbox(
            city_data,
            lat="sensor_latitude",
            lon="sensor_longitude",
            color="PM2.5",
            size="PM2.5",
            hover_name="sensor_name",
            hover_data=["PM2.5", "PM10", "NO2", "timestamp"],
            color_continuous_scale="RdYlGn_r",
            zoom=9,
            center={"lat": lat, "lon": lon},
            size_max=25
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ ALERTS TAB -------------------------
    def create_alerts_section(self, pollution_data, city, critical_pollutants):
        st.markdown('<div class="section-title">Real-Time Alerts & Notifications</div>', unsafe_allow_html=True)

        alerts = self.check_pollution_alerts(pollution_data)

        if not alerts:
            st.success("✅ No active alerts. Air quality is acceptable.")
        else:
            for alert in alerts:
                if alert["level"] == "CRITICAL":
                    st.error(alert["message"])
                elif alert["level"] == "WARNING":
                    st.warning(alert["message"])
                else:
                    st.info(alert["message"])

        st.markdown("**Send Email Alerts (EmailJS)**")
        with st.expander("Configure email alert", expanded=False):
            to_email = st.text_input("Recipient email")
            if st.button("Send current alerts via Email"):
                if not to_email.strip():
                    st.warning("Please enter a valid email address.")
                else:
                    subject = f"Air Quality Alerts - {city}"
                    lines = [f"Air Quality Alerts for {city} at {datetime.now():%Y-%m-%d %H:%M}", ""]
                    for a in alerts:
                        lines.append("- " + a["message"])
                    if not alerts:
                        lines.append("No critical alerts at this time.")
                    message = "\n".join(lines)
                    self.send_emailjs_alert(to_email, subject, message)
                    st.success("✅ Alert email triggered (check EmailJS logs).")

    # ------------------ REPORTS TAB ------------------------
    def create_reports_section(self, city, pollution_data, sources, probabilities):
        st.markdown('<div class="section-title">Reports & Statistical Overview</div>', unsafe_allow_html=True)

        top = st.columns(2)
        with top[0]:
            st.write("**City Snapshot**")
            st.write(f"- **City:** {city}")
            st.write(f"- **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.write("**Current Pollution Values:**")
            snapshot = {k: round(v, 2) for k, v in pollution_data.items() if isinstance(v, (int, float))}
            st.json(snapshot)

        with top[1]:
            st.write("**Source Contribution Table**")
            df_src = pd.DataFrame({
                "Source": sources,
                "Probability (%)": (probabilities * 100).round(2)
            })
            st.dataframe(df_src, use_container_width=True)

        st.write("**Last 7 Days - Daily Averages (PM2.5 / PM10 / NO2)**")
        city_data = self.df[self.df["city"] == city].copy()
        if city_data.empty:
            st.info("No historical data available.")
            return

        city_data["date"] = city_data["timestamp"].dt.date
        daily = city_data.groupby("date")[["PM2.5", "PM10", "NO2"]].mean().round(2)
        st.dataframe(daily.tail(7), use_container_width=True)

        csv_data = daily.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download Daily Averages (CSV)",
            data=csv_data,
            file_name=f"{city}_daily_averages.csv",
            mime="text/csv"
        )

    # ------------------ MAIN RUNNER ------------------------
    def run_dashboard(self):
        load_custom_css()
        self.load_data()
        if self.df is None or self.df.empty:
            st.error("❌ No data available to display.")
            return

        # INDIA THEME HEADER (LIKE YOUR IMAGE)
        st.markdown(
            """
            <div class="india-header">
                <div class="india-title">IN <b>AI-EnviroScan India</b></div>
                <div class="india-subtitle">
                    National Pollution Monitoring & Source Prediction System
                </div>
                <div class="india-tagline">
                    Real-time Air Quality Monitoring System
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        selected_city, lat, lon, pollution_data, input_method = self.create_sidebar_inputs()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 AI Analysis",
            "📈 Trends",
            "🗺️ Map",
            "🚨 Alerts",
            "📊 Reports"
        ])

        with tab1:
            critical_pollutants = self.create_prediction_section(selected_city, pollution_data)

        with tab2:
            self.create_trends_section(selected_city)

        with tab3:
            self.create_map_section(selected_city, lat, lon)

        with tab4:
            if "critical_pollutants" not in locals():
                critical_pollutants = []
            self.create_alerts_section(pollution_data, selected_city, critical_pollutants)

        with tab5:
            sources, probabilities = self.predict_pollution_source(pollution_data)
            self.create_reports_section(selected_city, pollution_data, sources, probabilities)

        st.markdown("---")
        st.markdown(
            f'<div class="footer-text">AI-EnviroScan India • Updated: '
            f'{datetime.now().strftime("%Y-%m-%d %H:%M")} • Corporate Analytics View</div>',
            unsafe_allow_html=True
        )


# ----------------------------------------------------------
# RUN APP
# ----------------------------------------------------------
if __name__ == "__main__":
    dashboard = FinalEnviroScanDashboard()
    dashboard.run_dashboard()
