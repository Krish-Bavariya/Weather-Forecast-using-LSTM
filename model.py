from sklearn.neural_network import MLPRegressor


def build_model(input_shape, use_tensorflow=True):
    if use_tensorflow:
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(20))  # 5 days x 4 parameters
        model.compile(optimizer="adam", loss="mse")
        return model

    # Fallback for Python versions where TensorFlow is unavailable.
    return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)