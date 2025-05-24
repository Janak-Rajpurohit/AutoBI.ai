# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from sklearn.preprocessing import MinMaxScaler
# from statsmodels.tsa.arima.model import ARIMA
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Dense
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from sklearn.metrics import mean_squared_error

# st.title("Time Series Forecasting (ARIMA, LSTM, GRU)")
 
# if "df" not in st.session_state:
#     st.warning("❗ Please upload a CSV file on the 'Upload' page first.")
#     st.stop()
# else:
#     df = st.session_state.df
#     st.dataframe(df)

# target_col = st.selectbox("Select column to forecast", df.select_dtypes(include=[np.number]).columns)
# forecast_steps = st.slider("Forecast steps", min_value=5, max_value=100, value=30)
# epochs = st.slider("Epochs for training", min_value=1, max_value=100, value=50)

# col_types = st.session_state.get("col_types", {})
# print(col_types)
# datetime_col = next((col for col, col_type in col_types.items() if col_type.lower() == "datetime"), None)

# # Handle datetime or generate synthetic index
# print(datetime_col, datetime_col in df.columns)
# print(df.columns)
# print(df)
# if datetime_col and datetime_col in df.columns:
#     st.success(f"Using '{datetime_col}' as datetime column.")
#     df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
#     df.dropna(subset=[datetime_col], inplace=True)
#     df.set_index(datetime_col, inplace=True)
#     inferred_freq = pd.infer_freq(df.index[:20]) or 'D'
#     x_original = df.index
#     x_forecast = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq=inferred_freq)[1:]
# else:
#     st.warning("⚠️ No datetime column found. Using numeric sequence as index.")
#     df.reset_index(drop=True, inplace=True)
#     x_original = list(range(len(df)))
#     x_forecast = list(range(len(df), len(df) + forecast_steps))

# if st.button("Forecast"):
#     series = df[target_col].dropna().values.astype('float32')

#     # Plot original series
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=x_original, y=series, mode='lines', name='Original Data'))
#     fig.update_layout(title='Time Series Data', xaxis_title='Index/Date', yaxis_title='Value')
#     st.plotly_chart(fig)

#     # ---------- ARIMA ----------
#     st.subheader("Forecast with ARIMA")
#     try:
#         model_arima = ARIMA(series, order=(5, 1, 0)).fit()
#         forecast_arima = model_arima.forecast(steps=forecast_steps)

#         fig_arima = go.Figure()
#         fig_arima.add_trace(go.Scatter(x=x_original, y=series, mode='lines', name='Original Data'))
#         fig_arima.add_trace(go.Scatter(x=x_forecast, y=forecast_arima, mode='lines', name='ARIMA Forecast', line=dict(dash='dot')))
#         fig_arima.update_layout(title="ARIMA Forecast", xaxis_title="Index/Date", yaxis_title="Value")
#         st.plotly_chart(fig_arima)

#         arima_error = mean_squared_error(series[-forecast_steps:], forecast_arima)
#         st.write(f"ARIMA Forecast MSE: {arima_error:.2f}")
#     except Exception as e:
#         st.error(f"ARIMA failed: {e}")

#     # ---------- Preprocess for LSTM/GRU ----------
#     scaler = MinMaxScaler()
#     series_scaled = scaler.fit_transform(series.reshape(-1, 1))
#     generator = TimeseriesGenerator(series_scaled, series_scaled, length=10, batch_size=1)

#     # ---------- LSTM ----------
#     st.subheader("Forecast with LSTM")
#     try:
#         lstm_model = Sequential()
#         lstm_model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(10, 1)))
#         lstm_model.add(LSTM(50, activation='relu'))
#         lstm_model.add(Dense(1))
#         lstm_model.compile(optimizer='adam', loss='mse')
#         lstm_model.fit(generator, epochs=epochs, verbose=0)

#         pred_input = series_scaled[-10:]
#         predictions = []
#         for _ in range(forecast_steps):
#             pred = lstm_model.predict(pred_input.reshape(1, 10, 1), verbose=0)
#             predictions.append(pred[0][0])
#             pred_input = np.append(pred_input[1:], pred, axis=0)
#         lstm_forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

#         fig_lstm = go.Figure()
#         fig_lstm.add_trace(go.Scatter(x=x_original, y=series, mode='lines', name='Original Data'))
#         fig_lstm.add_trace(go.Scatter(x=x_forecast, y=lstm_forecast, mode='lines', name='LSTM Forecast', line=dict(dash='dot')))
#         fig_lstm.update_layout(title="LSTM Forecast", xaxis_title="Index/Date", yaxis_title="Value")
#         st.plotly_chart(fig_lstm)

#         lstm_error = mean_squared_error(series[-forecast_steps:], lstm_forecast)
#         st.write(f"LSTM Forecast MSE: {lstm_error:.2f}")
#     except Exception as e:
#         st.error(f"LSTM failed: {e}")

#     # ---------- GRU ----------
#     st.subheader("Forecast with GRU")
#     try:
#         gru_model = Sequential()
#         gru_model.add(GRU(50, activation='relu', return_sequences=True, input_shape=(10, 1)))
#         gru_model.add(GRU(50, activation='relu'))
#         gru_model.add(Dense(1))
#         gru_model.compile(optimizer='adam', loss='mse')
#         gru_model.fit(generator, epochs=epochs, verbose=0)

#         pred_input = series_scaled[-10:]
#         predictions = []
#         for _ in range(forecast_steps):
#             pred = gru_model.predict(pred_input.reshape(1, 10, 1), verbose=0)
#             predictions.append(pred[0][0])
#             pred_input = np.append(pred_input[1:], pred, axis=0)
#         gru_forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

#         fig_gru = go.Figure()
#         fig_gru.add_trace(go.Scatter(x=x_original, y=series, mode='lines', name='Original Data'))
#         fig_gru.add_trace(go.Scatter(x=x_forecast, y=gru_forecast, mode='lines', name='GRU Forecast', line=dict(dash='dot')))
#         fig_gru.update_layout(title="GRU Forecast", xaxis_title="Index/Date", yaxis_title="Value")
#         st.plotly_chart(fig_gru)

#         gru_error = mean_squared_error(series[-forecast_steps:], gru_forecast)
#         st.write(f"GRU Forecast MSE: {gru_error:.2f}")
#     except Exception as e:
#         st.error(f"GRU failed: {e}")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error

st.title("Time Series Forecasting (ARIMA, LSTM, GRU)")

# Check if the DataFrame exists in session state
if "df" not in st.session_state:
    st.warning("❗ Please upload a CSV file on the 'Upload' page first.")
    st.stop()
else:
    df = st.session_state.df
    try:
        df.reset_index(inplace=True)
    except:
        pass
    st.dataframe(df,hide_index=True)

# Select the target column for forecasting
target_col = st.selectbox("Select column to forecast", df.select_dtypes(include=[np.number]).columns)
datetime_col = st.selectbox("Select the datetime column", st.session_state.date_columns)

# Select the aggregation method
aggregation_method = st.selectbox(
    "Select aggregation method for duplicate dates",
    options=["avg", "max", "min", "sum"],
    index=0
)

# Select the forecast steps and epochs
forecast_steps = st.slider("Forecast steps", min_value=5, max_value=100, value=30)
epochs = st.slider("Epochs for training", min_value=1, max_value=100, value=50)

# Handle datetime column
if datetime_col:
    st.success(f"Using '{datetime_col}' as the datetime column.")
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df.dropna(subset=[datetime_col], inplace=True)

    # Aggregate target column based on the datetime column
    if aggregation_method == "avg":
        df = df.groupby(datetime_col, as_index=False)[target_col].mean()
    elif aggregation_method == "max":
        df = df.groupby(datetime_col, as_index=False)[target_col].max()
    elif aggregation_method == "min":
        df = df.groupby(datetime_col, as_index=False)[target_col].min()
    elif aggregation_method == "sum":
        df = df.groupby(datetime_col, as_index=False)[target_col].sum()

    # Set the datetime column as the index
    df.set_index(datetime_col, inplace=True)
    inferred_freq = pd.infer_freq(df.index[:20]) or 'D'
    x_original = df.index
    x_forecast = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq=inferred_freq)[1:]
else:
    st.warning("⚠️ No datetime column found. Using a numeric sequence as the index.")
    df.reset_index(drop=True, inplace=True)
    x_original = list(range(len(df)))
    x_forecast = list(range(len(df), len(df) + forecast_steps))

# Forecasting button
if st.button("Forecast"):
    series = df[target_col].dropna().values.astype('float32')

    # Plot original series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_original, y=series, mode='lines', name='Original Data'))
    fig.update_layout(title='Time Series Data', xaxis_title='Index/Date', yaxis_title='Value')
    st.plotly_chart(fig)

    # ---------- ARIMA ----------
    st.subheader("Forecast with ARIMA")
    try:
        model_arima = ARIMA(series, order=(5, 1, 0)).fit()
        forecast_arima = model_arima.forecast(steps=forecast_steps)

        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=x_original, y=series, mode='lines', name='Original Data'))
        fig_arima.add_trace(go.Scatter(x=x_forecast, y=forecast_arima, mode='lines', name='ARIMA Forecast', line=dict(dash='dot')))
        fig_arima.update_layout(title="ARIMA Forecast", xaxis_title="Index/Date", yaxis_title="Value")
        st.plotly_chart(fig_arima)

        arima_error = mean_squared_error(series[-forecast_steps:], forecast_arima)
        st.write(f"ARIMA Forecast MSE: {arima_error:.2f}")
    except Exception as e:
        st.error(f"ARIMA failed: {e}")

    # ---------- Preprocess for LSTM/GRU ----------
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))
    generator = TimeseriesGenerator(series_scaled, series_scaled, length=10, batch_size=1)

    # ---------- LSTM ----------
    st.subheader("Forecast with LSTM")
    try:
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(10, 1)))
        lstm_model.add(LSTM(50, activation='relu'))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(generator, epochs=epochs, verbose=0)

        pred_input = series_scaled[-10:]
        predictions = []
        for _ in range(forecast_steps):
            pred = lstm_model.predict(pred_input.reshape(1, 10, 1), verbose=0)
            predictions.append(pred[0][0])
            pred_input = np.append(pred_input[1:], pred, axis=0)
        lstm_forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Scatter(x=x_original, y=series, mode='lines', name='Original Data'))
        fig_lstm.add_trace(go.Scatter(x=x_forecast, y=lstm_forecast, mode='lines', name='LSTM Forecast', line=dict(dash='dot')))
        fig_lstm.update_layout(title="LSTM Forecast", xaxis_title="Index/Date", yaxis_title="Value")
        st.plotly_chart(fig_lstm)

        lstm_error = mean_squared_error(series[-forecast_steps:], lstm_forecast)
        st.write(f"LSTM Forecast MSE: {lstm_error:.2f}")
    except Exception as e:
        st.error(f"LSTM failed: {e}")

    # ---------- GRU ----------
    st.subheader("Forecast with GRU")
    try:
        gru_model = Sequential()
        gru_model.add(GRU(50, activation='relu', return_sequences=True, input_shape=(10, 1)))
        gru_model.add(GRU(50, activation='relu'))
        gru_model.add(Dense(1))
        gru_model.compile(optimizer='adam', loss='mse')
        gru_model.fit(generator, epochs=epochs, verbose=0)

        pred_input = series_scaled[-10:]
        predictions = []
        for _ in range(forecast_steps):
            pred = gru_model.predict(pred_input.reshape(1, 10, 1), verbose=0)
            predictions.append(pred[0][0])
            pred_input = np.append(pred_input[1:], pred, axis=0)
        gru_forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        fig_gru = go.Figure()
        fig_gru.add_trace(go.Scatter(x=x_original, y=series, mode='lines', name='Original Data'))
        fig_gru.add_trace(go.Scatter(x=x_forecast, y=gru_forecast, mode='lines', name='GRU Forecast', line=dict(dash='dot')))
        fig_gru.update_layout(title="GRU Forecast", xaxis_title="Index/Date", yaxis_title="Value")
        st.plotly_chart(fig_gru)

        gru_error = mean_squared_error(series[-forecast_steps:], gru_forecast)
        st.write(f"GRU Forecast MSE: {gru_error:.2f}")
    except Exception as e:
        st.error(f"GRU failed: {e}")