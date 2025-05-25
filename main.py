import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import parser

# Import forecasting models
from models.linear_model import run_linear_regression_forecast
from models.arima_model import run_arima_forecast
from models.exp_smoothing_model import run_exp_smoothing_forecast


st.set_page_config(
    page_title="GA4 Forecast Tool",
    layout="wide",
    page_icon="📈"
)

st.title("📊 GA4 Forecasting Tool")


def try_parse_date(x):
    try:
        return parser.parse(x, dayfirst=False)  # Supports multiple formats, default ISO first
    except Exception:
        return pd.NaT


def load_data(file):
    try:
        df = pd.read_csv(file)
        
        # Strip and clean raw date strings
        df['date_raw'] = df['date'].astype(str).str.strip()
        
        # Show some raw date samples for debugging
        st.write("Unique sample date values:", df['date_raw'].dropna().unique()[:20])
        
        # Parse dates robustly
        df['date'] = df['date_raw'].apply(try_parse_date)
        
        if df['date'].isnull().any():
            bad_rows = df[df['date'].isnull()]
            st.error(f"Invalid dates found in these rows:\n{bad_rows[['date_raw', 'channel']].head(20)}")
            return None, None, None
        
        # Required columns
        required_cols = {'date', 'channel'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None, None, None
        
        # Metrics we support
        metric_candidates = ['sessions', 'orders', 'revenue']
        available_metrics = [m for m in metric_candidates if m in df.columns]
        if not available_metrics:
            st.error("No valid metric columns found (need sessions, orders, or revenue)")
            return None, None, None
        
        # Clean numeric metrics, drop rows with NaN metrics
        for metric in available_metrics:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=available_metrics)
        
        channels = df['channel'].astype(str).unique().tolist()
        
        return df, channels, available_metrics
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None


# Upload file section
uploaded_file = st.file_uploader(
    "Drag and drop your GA4 CSV file here",
    type=["csv"],
    help="Required columns: date (DD/MM/YYYY or ISO), channel. Metrics: sessions, orders, revenue"
)

if uploaded_file is not None:
    with st.expander("🔍 Raw File Preview", expanded=False):
        raw_df = pd.read_csv(uploaded_file)
        st.write("Columns found:", raw_df.columns.tolist())
        st.write("First 5 rows:", raw_df.head())
        uploaded_file.seek(0)  # Reset file pointer

    with st.spinner("🔍 Validating and processing data..."):
        df, channels, available_metrics = load_data(uploaded_file)

    if df is not None:
        st.success("✅ Data loaded successfully!")

        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("First Date", df['date'].min().strftime('%d/%m/%Y'))
        with col2:
            st.metric("Last Date", df['date'].max().strftime('%d/%m/%Y'))
        with col3:
            st.metric("Channels", len(channels))

        # Forecast configuration
        st.header("⚙️ Forecast Configuration")

        col1, col2 = st.columns(2)

        with col1:
            metric = st.selectbox(
                "Select metric to forecast:",
                available_metrics,
                format_func=lambda x: f"{x.title()} (Available)" if x in df.columns else x
            )

            model = st.radio(
                "Forecasting model:",
                ("Linear Regression", "ARIMA", "Exponential Smoothing")
            )

        with col2:
            forecast_days = st.slider(
                "Days to forecast:",
                min_value=7,
                max_value=180,
                value=30,
                step=7
            )

            channel_filter = st.selectbox(
                "Filter channel:",
                ["All"] + sorted(channels)
            )

        if st.button("🚀 Run Forecast"):
            try:
                filtered_df = df.copy()
                if channel_filter != "All":
                    filtered_df = filtered_df[filtered_df['channel'] == channel_filter]

                # Aggregate metric by date
                daily_data = filtered_df.groupby('date')[metric].sum()
                
                if len(daily_data) < 7:
                    raise ValueError("Need at least 7 days of data to forecast")

                daily_data.index = pd.to_datetime(daily_data.index)

                # Run the selected forecasting model
                if model == "Linear Regression":
                    forecast_df = run_linear_regression_forecast(daily_data.to_frame(), metric, forecast_days)
                elif model == "ARIMA":
                    forecast_df = run_arima_forecast(daily_data.to_frame(), metric, forecast_days)
                elif model == "Exponential Smoothing":
                    forecast_df = run_exp_smoothing_forecast(daily_data.to_frame(), metric, forecast_days)
                else:
                    raise ValueError("Unsupported model selected")

                # Plot historic + forecast
                st.header("📈 Forecast Results")

                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Historical data
                ax.plot(forecast_df.index, forecast_df[metric], label="Historical", marker='o')
                
                # Forecast data (if present)
                if 'forecast' in forecast_df.columns:
                    ax.plot(forecast_df.index, forecast_df['forecast'], label="Forecast", marker='o')

                ax.set_xlabel("Date")
                ax.set_ylabel(metric.title())
                ax.set_title(f"{metric.title()} Forecast - {channel_filter} ({model})")
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                ax.legend()

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Forecast failed: {str(e)}")

else:
    st.info("ℹ️ Please upload a GA4 CSV file with columns: date (DD/MM/YYYY or ISO), channel, and at least one metric column")
