import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
        font-size: 2.2rem;
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
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2933;
        padding-bottom: 6px;
        border-bottom: 1px solid #e1e7ee;
        margin-bottom: 12px;
    }
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
        font-size: 0.75rem;
        color: #8292a0;
        margin-top: 4px;
    }
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
            "PM10": {"good": 54, "moderate": 154, "unhealthy": 254, "hazardous": 424},
            "NO2": {"good": 40, "moderate": 100, "unhealthy": 360, "hazardous": 649}
        }
        self.df = None
        
        # Email configuration
        self.SMTP_SERVER = "smtp.gmail.com"
        self.SMTP_PORT = 587

    # ------------------------ GENERATE REPORT ----------------------
    def generate_report_html(self, city, pollution_data, sources, probabilities, alerts):
        """Generate HTML report for email"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Calculate AQI
        aqi = self.calculate_aqi(pollution_data)
        aqi_status = self.get_aqi_status(aqi)
        
        # Generate HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(90deg, #FFA756, #FFFFFF, #4CAF50); 
                          padding: 20px; border-radius: 10px; text-align: center; 
                          border: 2px solid #002060; margin-bottom: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 8px; }}
                .kpi-container {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
                .kpi {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .alert-critical {{ color: #d32f2f; background: #ffebee; padding: 10px; border-radius: 5px; margin: 5px 0; }}
                .alert-warning {{ color: #f57c00; background: #fff3e0; padding: 10px; border-radius: 5px; margin: 5px 0; }}
                .alert-notice {{ color: #1976d2; background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; 
                         color: #666; font-size: 12px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="color: #002060;">🌍 AI-EnviroScan India</h1>
                <h3>Air Quality Analysis Report</h3>
                <p>Generated: {current_time}</p>
            </div>
            
            <div class="section">
                <h2>📍 Location Summary</h2>
                <p><strong>City:</strong> {city}</p>
                <p><strong>Report Date:</strong> {current_time}</p>
                <p><strong>Overall AQI:</strong> <span style="font-size: 24px; font-weight: bold;">{aqi}</span> - {aqi_status}</p>
            </div>
            
            <div class="section">
                <h2>📊 Current Pollution Levels</h2>
                <div class="kpi-container">
        """
        
        # Add pollution KPIs
        for pollutant, value in pollution_data.items():
            if pollutant in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
                unit = "μg/m³" if pollutant != "CO" else "mg/m³"
                html_report += f"""
                <div class="kpi">
                    <h3>{pollutant}</h3>
                    <p style="font-size: 24px; font-weight: bold;">{value:.1f}</p>
                    <p>{unit}</p>
                    <p>{self.get_pollution_status(pollutant, value)}</p>
                </div>
                """
        
        html_report += """
                </div>
            </div>
            
            <div class="section">
                <h2>🤖 AI-Based Source Prediction</h2>
                <h3>Pollution Source Contributions:</h3>
                <table>
                    <tr><th>Source</th><th>Probability</th></tr>
        """
        
        # Add source predictions
        for source, prob in zip(sources, probabilities):
            percentage = prob * 100
            html_report += f"""
            <tr>
                <td>{source}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """
        
        html_report += """
                </table>
                <p><strong>Primary Source:</strong> """
        
        if len(sources) > 0:
            primary_idx = np.argmax(probabilities)
            html_report += f"{sources[primary_idx]} ({probabilities[primary_idx]*100:.1f}%)"
        
        html_report += """
                </p>
            </div>
            
            <div class="section">
                <h2>🚨 Active Alerts & Recommendations</h2>
        """
        
        if alerts:
            for alert in alerts:
                alert_class = ""
                if alert["level"] == "CRITICAL":
                    alert_class = "alert-critical"
                elif alert["level"] == "WARNING":
                    alert_class = "alert-warning"
                else:
                    alert_class = "alert-notice"
                
                html_report += f"""
                <div class="{alert_class}">
                    <strong>{alert["level"]}:</strong> {alert["message"]}
                </div>
                """
            
            html_report += "<h3>📋 Health Recommendations:</h3><ul>"
            if any(a["level"] == "CRITICAL" for a in alerts):
                html_report += "<li>❌ Avoid outdoor activities</li>"
                html_report += "<li>🏠 Stay indoors with windows closed</li>"
                html_report += "<li>😷 Use N95 mask if going outside</li>"
            elif any(a["level"] == "WARNING" for a in alerts):
                html_report += "<li>⚠️ Limit outdoor activities</li>"
                html_report += "<li>👵 Sensitive groups should stay indoors</li>"
                html_report += "<li>💨 Use air purifiers</li>"
            else:
                html_report += "<li>✅ Air quality is acceptable</li>"
                html_report += "<li>🌳 Good for outdoor activities</li>"
            html_report += "</ul>"
        else:
            html_report += "<p>✅ No active alerts. Air quality is within acceptable limits.</p>"
        
        html_report += f"""
            </div>
            
            <div class="footer">
                <p>🔬 This report was generated by AI-EnviroScan India - National Pollution Monitoring System</p>
                <p>📧 For questions or more information, contact: support@aienviroscanindia.com</p>
                <p>⚠️ This is an automated report. For medical emergencies, consult healthcare professionals.</p>
                <p>© {datetime.now().year} AI-EnviroScan India. All rights reserved.</p>
            </div>
        </body>
        </html>
        """
        
        return html_report

    def calculate_aqi(self, pollution_data):
        """Calculate Air Quality Index"""
        pm25 = pollution_data.get("PM2.5", 0)
        pm10 = pollution_data.get("PM10", 0)
        no2 = pollution_data.get("NO2", 0)
        
        max_value = max(pm25, pm10, no2/2)
        aqi = min(max_value * 2, 500)
        return int(aqi)

    def get_aqi_status(self, aqi):
        """Get AQI status"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def get_pollution_status(self, pollutant, value):
        """Get pollution status for a specific pollutant"""
        if pollutant in self.POLLUTION_THRESHOLDS:
            th = self.POLLUTION_THRESHOLDS[pollutant]
            if value >= th["hazardous"]:
                return "🔴 HAZARDOUS"
            elif value >= th["unhealthy"]:
                return "🟠 Unhealthy"
            elif value >= th["moderate"]:
                return "🟡 Moderate"
            else:
                return "✅ Good"
        return ""

    # ------------------------ SEND EMAIL REPORT ----------------------
    def send_report_email(self, to_email, sender_email, sender_password, city, pollution_data, 
                         sources, probabilities, alerts):
        """Send complete report via email"""
        try:
            # Generate HTML report
            html_report = self.generate_report_html(city, pollution_data, sources, 
                                                   probabilities, alerts)
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"AI-EnviroScan India <{sender_email}>"
            msg['To'] = to_email
            msg['Subject'] = f"🌍 AI-EnviroScan Report - Air Quality Analysis for {city}"
            
            # Add text version
            text_content = f"""
            AI-EnviroScan India - Air Quality Report
            
            City: {city}
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            
            Current Pollution Levels:
            PM2.5: {pollution_data.get('PM2.5', 0):.1f} μg/m³
            PM10: {pollution_data.get('PM10', 0):.1f} μg/m³
            NO2: {pollution_data.get('NO2', 0):.1f} μg/m³
            
            Primary Pollution Source: {sources[0] if sources else 'Unknown'}
            
            Active Alerts: {len(alerts)}
            
            Please view the HTML version for complete details.
            
            ---
            AI-EnviroScan India
            National Pollution Monitoring System
            """
            
            # Attach both text and HTML versions
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_report, 'html'))
            
            # Connect to SMTP server and send
            server = smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT)
            server.starttls()
            server.login(sender_email, sender_password)
            
            text = msg.as_string()
            server.sendmail(sender_email, to_email, text)
            server.quit()
            
            return True, "✅ Report sent successfully!"
            
        except Exception as e:
            return False, f"❌ Error sending report: {str(e)}"

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
        for date in dates:
            for city, zone in zip(cities, zones):
                rows.append({
                    "timestamp": date,
                    "city": city,
                    "zone": zone,
                    "sensor_name": f"Sensor_{city}_{np.random.randint(1,5)}",
                    "PM2.5": np.random.uniform(30, 200),
                    "PM10": np.random.uniform(60, 300),
                    "NO2": np.random.uniform(20, 120),
                    "SO2": np.random.uniform(5, 50),
                    "CO": np.random.uniform(0.5, 5.0),
                    "O3": np.random.uniform(20, 100),
                    "temperature_c": np.random.uniform(15, 35),
                    "humidity": np.random.uniform(40, 90),
                    "wind_speed": np.random.uniform(5, 25),
                    "sensor_latitude": 20 + np.random.uniform(-5, 5),
                    "sensor_longitude": 78 + np.random.uniform(-5, 5)
                })
        self.df = pd.DataFrame(rows)

    # ---------------------- AI PREDICTION ------------------
    def predict_pollution_source(self, pollution_data):
        """Predict pollution sources based on pollutant ratios"""
        sources = ["Vehicle Emissions", "Industrial", "Construction Dust", "Agricultural Burning"]
        
        # Simple heuristic based on pollutant ratios
        pm_ratio = pollution_data.get("PM2.5", 0) / max(pollution_data.get("PM10", 1), 1)
        no2_level = pollution_data.get("NO2", 0)
        
        if pm_ratio > 0.5 and no2_level > 60:
            probs = np.array([0.6, 0.2, 0.1, 0.1])
        elif pm_ratio > 0.5:
            probs = np.array([0.3, 0.1, 0.1, 0.5])
        elif no2_level > 80:
            probs = np.array([0.4, 0.5, 0.05, 0.05])
        else:
            probs = np.array([0.25, 0.25, 0.25, 0.25])
            
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
                "PM10": st.sidebar.slider("PM10 (μg/m³)", 0, 600, 150),
                "NO2": st.sidebar.slider("NO2 (μg/m³)", 0, 200, 40),
                "SO2": st.sidebar.slider("SO2 (μg/m³)", 0, 100, 15),
                "CO": st.sidebar.slider("CO (mg/m³)", 0.0, 10.0, 1.2),
                "O3": st.sidebar.slider("O3 (μg/m³)", 0, 200, 45),
            }
            st.sidebar.subheader("📍 Manual Location")
            lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
            lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f")
            
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
                    "PM10": city_data["PM10"].mean(),
                    "NO2": city_data["NO2"].mean(),
                    "SO2": city_data.get("SO2", pd.Series([0]*len(city_data))).mean(),
                    "CO": city_data.get("CO", pd.Series([0]*len(city_data))).mean(),
                    "O3": city_data.get("O3", pd.Series([0]*len(city_data))).mean(),
                    "temperature_c": city_data.get("temperature_c", pd.Series([0]*len(city_data))).mean(),
                    "humidity": city_data.get("humidity", pd.Series([0]*len(city_data))).mean(),
                    "wind_speed": city_data.get("wind_speed", pd.Series([0]*len(city_data))).mean()
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
                    "PM10": city_data["PM10"].mean(),
                    "NO2": city_data["NO2"].mean(),
                    "SO2": city_data.get("SO2", pd.Series([0]*len(city_data))).mean(),
                    "CO": city_data.get("CO", pd.Series([0]*len(city_data))).mean(),
                    "O3": city_data.get("O3", pd.Series([0]*len(city_data))).mean(),
                    "temperature_c": city_data.get("temperature_c", pd.Series([0]*len(city_data))).mean(),
                    "humidity": city_data.get("humidity", pd.Series([0]*len(city_data))).mean(),
                    "wind_speed": city_data.get("wind_speed", pd.Series([0]*len(city_data))).mean()
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

        return critical_pollutants, sources, probs

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
        df_filtered = df_city[df_city["timestamp"] >= min_date]

        if df_filtered.empty:
            st.warning("No data in the selected time range.")
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

    # ------------------ SEND REPORT SECTION -----------------
    def create_report_email_section(self, city, pollution_data, sources, probabilities, alerts):
        st.markdown('<div class="section-title">📧 Send Complete Analysis Report</div>', unsafe_allow_html=True)
        
        st.info("Send a **complete HTML report** of this analysis to any email address.")
        
        with st.expander("📤 Configure Email Report", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📧 Your Email Credentials")
                st.caption("(Used to send the report - not stored anywhere)")
                sender_email = st.text_input("Your Email Address", 
                                            value="emvs2005@gmail.com",
                                            placeholder="your_email@gmail.com")
                sender_password = st.text_input("Your Email Password", 
                                               type="password",
                                               placeholder="Enter your email password")
                
                st.markdown("---")
                st.info("**For Gmail users with 2FA:**")
                st.markdown("""
                1. Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
                2. Generate an "App Password"
                3. Use that instead of your regular password
                """)
            
            with col2:
                st.subheader("📨 Recipient Details")
                recipient_email = st.text_input("Recipient Email Address",
                                               placeholder="recipient@example.com")
                
                st.markdown("---")
                st.subheader("📊 Report Preview")
                st.markdown(f"""
                **Report will include:**
                - ✅ City: {city}
                - 📊 Pollution Levels (PM2.5, PM10, NO2, etc.)
                - 🤖 AI Source Analysis
                - 🚨 Active Alerts ({len(alerts)} alerts)
                - 💡 Health Recommendations
                - 📅 Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                """)
            
            # Send button
            if st.button("📨 Send Complete Report", type="primary", use_container_width=True):
                if not all([sender_email, sender_password, recipient_email]):
                    st.warning("⚠️ Please fill in all email fields.")
                else:
                    with st.spinner("📤 Generating and sending report..."):
                        success, message = self.send_report_email(
                            recipient_email, sender_email, sender_password,
                            city, pollution_data, sources, probabilities,
                            alerts
                        )
                        
                        if success:
                            st.success(message)
                            st.balloons()
                        else:
                            st.error(message)
                            
                            # Show troubleshooting tips
                            if "authentication" in message.lower():
                                st.info("""
                                **Authentication Issues:**
                                1. Check your email and password
                                2. For Gmail, allow "Less Secure Apps" or use App Password
                                3. Make sure 2-factor authentication is properly configured
                                """)

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
        
        # INDIA THEME HEADER
        st.markdown(
            """
            <div class="india-header">
                <div class="india-title">AI-EnviroScan India</div>
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
            "📧 Send Report",
            "📊 Reports"
        ])

        with tab1:
            critical_pollutants, sources, probabilities = self.create_prediction_section(selected_city, pollution_data)
            alerts = self.check_pollution_alerts(pollution_data)

        with tab2:
            self.create_trends_section(selected_city)

        with tab3:
            self.create_map_section(selected_city, lat, lon)

        with tab4:
            if "critical_pollutants" not in locals():
                critical_pollutants = []
            if "alerts" not in locals():
                alerts = []
            if "sources" not in locals():
                sources, probabilities = self.predict_pollution_source(pollution_data)
            
            self.create_report_email_section(
                selected_city, pollution_data, 
                sources, probabilities, alerts
            )

        with tab5:
            if "sources" not in locals():
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