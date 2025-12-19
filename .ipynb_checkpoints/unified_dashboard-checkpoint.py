# unified_dashboard.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AIEnviroScanDashboard:
    def __init__(self):
        self.DASHBOARD_TITLE = "🌍 AI-EnviroScan Real-Time Dashboard"
        self.POLLUTION_THRESHOLDS = {
            'PM2.5': {'good': 12, 'moderate': 35, 'unhealthy': 55, 'hazardous': 150},
            'PM10': {'good': 54, 'moderate': 154, 'unhealthy': 254, 'hazardous': 424},
            'NO2': {'good': 40, 'moderate': 100, 'unhealthy': 360, 'hazardous': 649}
        }
        self.df = None
        self.model_artifacts = None
        self.has_predictions = False
        self.pred_df = None
        
    def load_data(self):
        """Load all required data and models"""
        print("📥 LOADING DATA AND MODELS...")
        try:
            # Load data
            self.df = pd.read_csv('data/pollution_data_comprehensive.csv')
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Load models
            self.model_artifacts = joblib.load('models/pollution_source_model_artifacts.joblib')
            
            # Load prediction data if available
            try:
                self.pred_df = pd.read_csv('exports/prediction_accuracy_report.csv')
                self.has_predictions = True
            except:
                self.has_predictions = False
            
            print("✅ All data loaded successfully!")
            print(f"📊 Dataset: {self.df.shape[0]:,} records")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Create sample data for demonstration
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data if real data isn't available"""
        print("📝 Creating sample data for demonstration...")
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
        sensors = ['Sensor_A', 'Sensor_B', 'Sensor_C', 'Sensor_D']
        
        sample_data = []
        for date in dates:
            for sensor in sensors:
                sample_data.append({
                    'timestamp': date,
                    'sensor_name': sensor,
                    'sensor_latitude': 18.52 + np.random.uniform(-0.1, 0.1),
                    'sensor_longitude': 73.85 + np.random.uniform(-0.1, 0.1),
                    'PM2.5': np.random.uniform(10, 150),
                    'PM10': np.random.uniform(20, 300),
                    'NO2': np.random.uniform(5, 100),
                    'SO2': np.random.uniform(2, 50),
                    'CO': np.random.uniform(0.5, 5),
                    'O3': np.random.uniform(10, 80),
                    'temperature_c': np.random.uniform(15, 35),
                    'humidity': np.random.uniform(40, 90),
                    'wind_speed': np.random.uniform(0, 15),
                    'area_type': np.random.choice(['Industrial', 'Residential', 'Commercial', 'Traffic'])
                })
        
        self.df = pd.DataFrame(sample_data)
        
        # Create sample model artifacts
        self.model_artifacts = {
            'metadata': {
                'best_model': 'Random Forest',
                'test_accuracy': 0.85,
                'training_date': '2024-01-01'
            }
        }
        
        print("✅ Sample data created successfully!")
    
    def create_header(self):
        """Create dashboard header"""
        header_html = """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; 
                    border-radius: 10px; 
                    color: white; 
                    text-align: center;
                    margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🌍 AI-EnviroScan</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Real-Time Pollution Monitoring & Source Prediction System</p>
            <p style="margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.8;">
                Powered by Machine Learning | Pune, India | Live Updates Every 15 Minutes
            </p>
        </div>
        """
        display(HTML(header_html))
    
    def create_metrics(self):
        """Create key metrics dashboard"""
        current_data = self.df[self.df['timestamp'] >= (self.df['timestamp'].max() - timedelta(hours=24))]
        
        metrics = {
            'Active Sensors': self.df['sensor_name'].nunique(),
            'Total Readings': f"{len(self.df):,}",
            'Current PM2.5': f"{current_data['PM2.5'].mean():.1f} μg/m³",
            'Current PM10': f"{current_data['PM10'].mean():.1f} μg/m³",
            'Data Coverage': f"{(self.df['timestamp'].max() - self.df['timestamp'].min()).days} days",
            'Model Accuracy': f"{self.model_artifacts['metadata']['test_accuracy']:.1%}"
        }
        
        metrics_html = """
        <div style="display: grid; 
                    grid-template-columns: repeat(3, 1fr); 
                    gap: 15px; 
                    margin-bottom: 20px;">
        """
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A8EAE', '#57A773']
        for i, (key, value) in enumerate(metrics.items()):
            metrics_html += f"""
            <div style="background-color: {colors[i]}; 
                        padding: 20px; 
                        border-radius: 8px; 
                        color: white; 
                        text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 0.9em; opacity: 0.9;">{key}</div>
                <div style="font-size: 1.8em; font-weight: bold; margin: 10px 0;">{value}</div>
            </div>
            """
        
        metrics_html += "</div>"
        display(HTML(metrics_html))
    
    def create_alerts(self):
        """Create pollution alert system"""
        alerts = self.check_pollution_alerts()
        
        if alerts:
            alerts_html = """
            <div style="background-color: #fff3cd; 
                        border: 1px solid #ffeaa7; 
                        border-radius: 8px; 
                        padding: 15px; 
                        margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0; color: #856404;">🚨 ACTIVE POLLUTION ALERTS</h3>
            """
            
            for alert in alerts:
                alerts_html += f"""
                <div style="background-color: {'#f8d7da' if 'CRITICAL' in alert['level'] else '#fff3cd' if 'WARNING' in alert['level'] else '#d1ecf1'}; 
                            padding: 10px; 
                            margin: 5px 0; 
                            border-radius: 5px; 
                            border-left: 4px solid {'#dc3545' if 'CRITICAL' in alert['level'] else '#ffc107' if 'WARNING' in alert['level'] else '#17a2b8'};">
                    <strong>{alert['level']}</strong> {alert['message']}<br>
                    <small>Sensor: {alert['sensor']} | Time: {self.df['timestamp'].max().strftime('%H:%M')}</small>
                </div>
                """
            
            alerts_html += "</div>"
        else:
            alerts_html = """
            <div style="background-color: #d4edda; 
                        border: 1px solid #c3e6cb; 
                        border-radius: 8px; 
                        padding: 15px; 
                        margin-bottom: 20px;">
                <h3 style="margin: 0; color: #155724;">✅ ALL SYSTEMS NORMAL</h3>
                <p style="margin: 5px 0 0 0; color: #155724;">All pollution levels within acceptable limits</p>
            </div>
            """
        
        display(HTML(alerts_html))
    
    def check_pollution_alerts(self):
        """Check for pollution threshold violations"""
        alerts = []
        current_data = self.df[self.df['timestamp'] >= (self.df['timestamp'].max() - timedelta(hours=1))]
        
        for pollutant, thresholds in self.POLLUTION_THRESHOLDS.items():
            if pollutant in current_data.columns:
                max_value = current_data[pollutant].max()
                
                if max_value >= thresholds['hazardous']:
                    alerts.append({
                        'level': '🔴 CRITICAL',
                        'message': f'{pollutant} at {max_value:.1f} μg/m³ - HAZARDOUS levels!',
                        'sensor': current_data.loc[current_data[pollutant].idxmax(), 'sensor_name']
                    })
                elif max_value >= thresholds['unhealthy']:
                    alerts.append({
                        'level': '🟠 WARNING', 
                        'message': f'{pollutant} at {max_value:.1f} μg/m³ - Unhealthy levels',
                        'sensor': current_data.loc[current_data[pollutant].idxmax(), 'sensor_name']
                    })
        
        return alerts
    
    def create_controls(self):
        """Create interactive controls"""
        sensor_dropdown = widgets.Dropdown(
            options=['All Sensors'] + sorted(self.df['sensor_name'].unique().tolist()),
            value='All Sensors',
            description='Sensor:',
            style={'description_width': 'initial'}
        )
        
        pollutant_dropdown = widgets.Dropdown(
            options=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
            value='PM2.5',
            description='Pollutant:',
            style={'description_width': 'initial'}
        )
        
        date_range_slider = widgets.SelectionRangeSlider(
            options=[(date.strftime('%Y-%m-%d'), date) for date in sorted(self.df['timestamp'].dt.date.unique())],
            index=(0, len(self.df['timestamp'].dt.date.unique()) - 1),
            description='Date Range:',
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        
        return sensor_dropdown, pollutant_dropdown, date_range_slider
    
    def create_pollution_trends(self, sensor='All Sensors', pollutant='PM2.5'):
        """Create pollution trend visualization"""
        if sensor == 'All Sensors':
            plot_data = self.df
            title_sensor = 'All Sensors'
        else:
            plot_data = self.df[self.df['sensor_name'] == sensor]
            title_sensor = sensor
        
        hourly_data = plot_data.set_index('timestamp').resample('H')[pollutant].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_data['timestamp'],
            y=hourly_data[pollutant],
            mode='lines',
            name=f'{pollutant} Trend',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        # Add threshold lines
        thresholds = self.POLLUTION_THRESHOLDS.get(pollutant, {})
        colors = {'good': 'green', 'moderate': 'yellow', 'unhealthy': 'orange', 'hazardous': 'red'}
        
        for level, value in thresholds.items():
            if value > 0:
                fig.add_hline(
                    y=value,
                    line_dash="dash",
                    line_color=colors.get(level, 'gray'),
                    annotation_text=f"{level.title()} ({value} μg/m³)",
                    annotation_position="right"
                )
        
        fig.update_layout(
            title=f'{pollutant} Pollution Trend - {title_sensor}',
            xaxis_title='Time',
            yaxis_title=f'{pollutant} Concentration (μg/m³)',
            height=400,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_sensor_comparison(self, pollutant='PM2.5'):
        """Compare pollution levels across sensors"""
        sensor_stats = (
            self.df.groupby('sensor_name')[pollutant]
            .agg(['mean', 'std', 'max', 'min'])
            .reset_index()
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sensor_stats['sensor_name'],
            y=sensor_stats['mean'],
            name=f'{pollutant} Mean',
            error_y=dict(type='data', array=sensor_stats['std'], visible=True)
        ))
        
        fig.add_trace(go.Scatter(
            x=sensor_stats['sensor_name'],
            y=sensor_stats['max'],
            mode='lines+markers',
            name=f'{pollutant} Max',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title=f'{pollutant} Levels by Sensor (Average ± Standard Deviation)',
            xaxis_title='Sensor Name',
            yaxis_title=f'{pollutant} Concentration (μg/m³)',
            height=500,
            template='plotly_white'
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_geographic_overview(self):
        """Create geographic distribution of pollution"""
        sensor_avg = self.df.groupby('sensor_name').agg({
            'sensor_latitude': 'first',
            'sensor_longitude': 'first',
            'PM2.5': 'mean',
            'PM10': 'mean',
            'area_type': 'first'
        }).reset_index()
        
        fig = px.scatter_mapbox(
            sensor_avg,
            lat="sensor_latitude",
            lon="sensor_longitude",
            color="PM2.5",
            size="PM2.5",
            hover_name="sensor_name",
            hover_data=["PM10", "area_type"],
            color_continuous_scale="Viridis",
            size_max=20,
            zoom=10,
            title="Pollution Sensor Network - Geographic Distribution"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=500
        )
        
        return fig
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        print("🚀 STARTING AI-ENVIROSCAN DASHBOARD...")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Create dashboard components
        self.create_header()
        self.create_metrics()
        self.create_alerts()
        
        # Create controls
        sensor_dropdown, pollutant_dropdown, date_range_slider = self.create_controls()
        
        print("🔧 DASHBOARD CONTROLS:")
        display(widgets.HBox([sensor_dropdown, pollutant_dropdown]))
        display(date_range_slider)
        
        # Create output area for dynamic updates
        output = widgets.Output()
        
        def update_dashboard(change):
            with output:
                clear_output(wait=True)
                selected_sensor = sensor_dropdown.value
                selected_pollutant = pollutant_dropdown.value
                
                print(f"📊 Displaying: {selected_sensor} | Pollutant: {selected_pollutant}")
                print("─" * 50)
                
                # Update trends
                print("📈 POLLUTION TRENDS")
                trend_fig = self.create_pollution_trends(selected_sensor, selected_pollutant)
                trend_fig.show()
                
                # Update comparison
                print("\n📊 SENSOR COMPARISON")
                comparison_fig = self.create_sensor_comparison(selected_pollutant)
                comparison_fig.show()
                
                # Show geographic overview for all sensors
                if selected_sensor == 'All Sensors':
                    print("\n🗺️ GEOGRAPHIC OVERVIEW")
                    geo_fig = self.create_geographic_overview()
                    geo_fig.show()
        
        # Link controls to update function
        sensor_dropdown.observe(update_dashboard, names='value')
        pollutant_dropdown.observe(update_dashboard, names='value')
        date_range_slider.observe(update_dashboard, names='value')
        
        print("🎯 INTERACTIVE DASHBOARD READY:")
        print("Change the controls above to update the visualizations below:")
        display(output)
        
        # Initial display
        update_dashboard(None)
        
        print("\n🎉 DASHBOARD SUCCESSFULLY DEPLOYED!")
        print("   Use the controls above to explore the data")

# For direct execution
if __name__ == "__main__":
    dashboard = AIEnviroScanDashboard()
    dashboard.run_dashboard()