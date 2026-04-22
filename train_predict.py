import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from data_api import get_coordinates, get_weather_data
from model import build_model


def train_and_predict(city):

    lat, lon = get_coordinates(city)
    df = get_weather_data(lat, lon)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    joblib.dump(scaler, "scaler.save")

    X, y = [], []
    time_steps = 30

    for i in range(len(scaled) - time_steps - 5):
        X.append(scaled[i:i+time_steps])
        y.append(scaled[i+time_steps:i+time_steps+5].flatten())

    X = np.array(X)
    y = np.array(y)

    try:
        import tensorflow  # noqa: F401

        use_tensorflow = True
    except ModuleNotFoundError:
        use_tensorflow = False

    model = build_model((time_steps, 4), use_tensorflow=use_tensorflow)

    last_30 = scaled[-30:]
    X_input = np.expand_dims(last_30, axis=0)

    if use_tensorflow:
        model.fit(X, y, epochs=25, batch_size=16, verbose=0)
        model.save("weather_lstm.h5")
        pred = model.predict(X_input, verbose=0)
    else:
        X_flat = X.reshape(X.shape[0], -1)
        X_input_flat = X_input.reshape(1, -1)
        model.fit(X_flat, y)
        joblib.dump(model, "weather_mlp_model.save")
        pred = model.predict(X_input_flat)

    pred = pred.reshape(5, 4)

    pred_real = scaler.inverse_transform(pred)

    return pred_real