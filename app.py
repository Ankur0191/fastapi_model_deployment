from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the model and scaler
try:
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    raise

# Define request model
class WeatherData(BaseModel):
    temperature: float
    solar_radiation: float
    wind_speed: float
    site_encoded: int

# Feature engineering function (same as in the training code)
def add_lag_and_rolling_features(data):
    # Assuming data is a DataFrame
    lags = [1, 3]
    windows = [3, 7]

    for col in ['temperature', 'solar_radiation', 'wind_speed']:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)

        for window in windows:
            data[f'{col}_rollmean_{window}'] = data[col].rolling(window).mean()
            data[f'{col}_rollstd_{window}'] = data[col].rolling(window).std()

    return data

# Prediction endpoint
@app.post("/predict/")
async def predict(data: WeatherData):
    try:
        # Prepare the input data as a DataFrame
        input_data = {
            "temperature": [data.temperature],
            "solar_radiation": [data.solar_radiation],
            "wind_speed": [data.wind_speed],
            "site_encoded": [data.site_encoded]
        }
        input_df = pd.DataFrame(input_data)
        
        # Apply feature engineering (add lags and rolling features)
        input_df = add_lag_and_rolling_features(input_df)

        # Fill missing values from lag/rolling
        input_df = input_df.fillna(0)  # Or another method to handle NaNs

        # Scale the input data using the same scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)

        # Return prediction
        return {"predicted_energy": prediction[0]}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": str(e)}
