import requests
import pandas as pd
from datetime import datetime, timedelta

def get_coordinates(city):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    r = requests.get(url).json()

    if "results" not in r:
        raise ValueError("City not found")

    lat = r["results"][0]["latitude"]
    lon = r["results"][0]["longitude"]
    return lat, lon


def get_weather_data(lat, lon):

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,relative_humidity_2m_mean,"
        f"wind_speed_10m_max,cloud_cover_mean"
        f"&timezone=auto"
    )

    data = requests.get(url).json()
    daily = data["daily"]

    df = pd.DataFrame({
        "temperature": daily["temperature_2m_mean"],
        "humidity": daily["relative_humidity_2m_mean"],
        "windspeed": daily["wind_speed_10m_max"],
        "cloudcover": daily["cloud_cover_mean"]
    })

    df.dropna(inplace=True)
    return df