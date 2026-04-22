import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Weather LSTM Forecast", page_icon="🌦️", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🌦️ LSTM Weather Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Train on last 365 days and predict next 5 days.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Forecast Settings")
    city = st.text_input("City", "Mumbai").strip()
    mode = st.radio(
        "Mode",
        ["Live forecast (recommended)", "ML forecast (your model)"],
        index=0,
    )
    run_btn = st.button("Predict Weather", use_container_width=True)
    st.caption("Data source: Open-Meteo APIs (similar to Google providers)")

if run_btn:
    if not city:
        st.warning("Please enter a city name.")
    else:
        try:
            if mode.startswith("Live"):
                from data_api import get_coordinates, get_live_forecast

                with st.spinner("Fetching live forecast..."):
                    lat, lon = get_coordinates(city)
                    live_df = get_live_forecast(lat, lon, days=5)

                pred = live_df[["temperature", "humidity", "windspeed", "cloudcover"]].to_numpy()
                day_labels = [str(d) for d in live_df["date"].tolist()]
            else:
                from train_predict import train_and_predict

                with st.spinner("Training model and generating forecast..."):
                    pred = train_and_predict(city)
                day_labels = [f"Day {d}" for d in [1, 2, 3, 4, 5]]
        except ModuleNotFoundError as exc:
            if "tensorflow" in str(exc).lower():
                st.error(
                    "TensorFlow is not installed for this Python version. "
                    "Please use Python 3.10-3.12, then run install_requirements.ps1 again."
                )
            else:
                st.error(f"Missing module: {exc}")
        except Exception as exc:
            st.error(f"Could not generate forecast: {exc}")
        else:
            st.success(f"Forecast generated for {city.title()}")

            days = [1, 2, 3, 4, 5]
            columns = ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Cloud Cover (%)"]
            forecast_df = pd.DataFrame(pred, columns=columns, index=day_labels).round(2)

            metric_cols = st.columns(4)
            metric_cols[0].metric("Avg Temp", f"{forecast_df['Temperature (°C)'].mean():.1f} °C")
            metric_cols[1].metric("Avg Humidity", f"{forecast_df['Humidity (%)'].mean():.1f} %")
            metric_cols[2].metric("Avg Wind", f"{forecast_df['Wind Speed (km/h)'].mean():.1f} km/h")
            metric_cols[3].metric("Avg Clouds", f"{forecast_df['Cloud Cover (%)'].mean():.1f} %")

            st.subheader("5-Day Forecast Table")
            st.dataframe(forecast_df, use_container_width=True)

            charts = [
                ("Temperature", "Temperature (°C)", pred[:, 0], "#ff6b6b"),
                ("Humidity", "Humidity (%)", pred[:, 1], "#339af0"),
                ("Wind Speed", "Wind Speed (km/h)", pred[:, 2], "#40c057"),
                ("Cloud Cover", "Cloud Cover (%)", pred[:, 3], "#845ef7"),
            ]
            tabs = st.tabs([item[0] for item in charts])

            for tab, (title, ylabel, values, color) in zip(tabs, charts):
                with tab:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(days, values, marker="o", linewidth=2.5, color=color)
                    ax.set_xticks(days)
                    if mode.startswith("Live"):
                        ax.set_xlabel("Next 5 Days (dates in table)")
                    else:
                        ax.set_xlabel("Next 5 Days")
                    ax.set_ylabel(ylabel)
                    ax.set_title(f"{title} Forecast for {city.title()}")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)