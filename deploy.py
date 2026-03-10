"""
deploy.py — FastAPI Deployment for House Price Prediction Model

This file loads the trained Keras model and scaler, then exposes
a REST API endpoint that accepts house features and returns
the predicted price.

Run with:
    uvicorn deploy:app --reload

Then open: http://127.0.0.1:8000/docs  (auto-generated Swagger UI)
"""

import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


# ─────────────────────────────────────────────
# 1. Load the trained model and scaler from disk
#    These were saved after training in the notebook
# ─────────────────────────────────────────────
with open("model_weights.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)


# ─────────────────────────────────────────────
# 2. Define the input schema using Pydantic
#    Each field represents one house feature.
#    FastAPI uses this to validate incoming JSON requests.
# ─────────────────────────────────────────────
class HouseFeatures(BaseModel):
    bedrooms: float        # Number of bedrooms
    bathrooms: float       # Number of bathrooms
    sqft_living: float     # Interior living space (square feet)
    sqft_lot: float        # Land space (square feet)
    floors: float          # Number of floors
    waterfront: int        # 1 = waterfront view, 0 = no waterfront
    view: int              # View quality index (0–4)
    condition: int         # Condition index (1–5)
    grade: int             # Construction quality (1–13)
    sqft_above: float      # Square footage above ground
    sqft_basement: float   # Square footage below ground
    yr_built: int          # Year the house was built
    yr_renovated: int      # Year of last renovation (0 if never)
    zipcode: int           # ZIP code of the property
    lat: float             # Latitude coordinate
    long: float            # Longitude coordinate
    sqft_living15: float   # Living area of nearest 15 neighbors
    sqft_lot15: float      # Lot area of nearest 15 neighbors
    month: int             # Month of sale (1–12)
    year: int              # Year of sale


# ─────────────────────────────────────────────
# 3. Create the FastAPI application instance
# ─────────────────────────────────────────────
app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house prices in King County using a trained Keras Neural Network.",
    version="1.0.0"
)


# ─────────────────────────────────────────────
# 4. Health check endpoint
#    GET /  → confirms the API is running
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "House Price Prediction API is running. Visit /docs for usage."}


# ─────────────────────────────────────────────
# 5. Prediction endpoint
#    POST /predict → accepts house features, returns predicted price
# ─────────────────────────────────────────────
@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Accepts a JSON body with house features.
    Returns the predicted house price in US dollars.

    Example request body:
    {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "yr_built": 1990,
        "yr_renovated": 0,
        "zipcode": 98178,
        "lat": 47.5112,
        "long": -122.257,
        "sqft_living15": 1340,
        "sqft_lot15": 5650,
        "month": 6,
        "year": 2015
    }
    """

    # Convert the Pydantic model to a list in the same column order used during training
    feature_values = [
        features.bedrooms,
        features.bathrooms,
        features.sqft_living,
        features.sqft_lot,
        features.floors,
        features.waterfront,
        features.view,
        features.condition,
        features.grade,
        features.sqft_above,
        features.sqft_basement,
        features.yr_built,
        features.yr_renovated,
        features.zipcode,
        features.lat,
        features.long,
        features.sqft_living15,
        features.sqft_lot15,
        features.month,
        features.year,
    ]

    # Reshape to 2D array: (1 sample, 20 features) — required by sklearn scaler
    input_array = np.array(feature_values).reshape(1, -1)

    # Scale the input using the same scaler fitted during training
    # This is critical — the model was trained on scaled data
    input_scaled = scaler.transform(input_array)

    # Run the prediction through the neural network
    prediction = model.predict(input_scaled)

    # Extract the single predicted value and round to 2 decimal places
    predicted_price = float(prediction[0][0])

    return {
        "predicted_price_usd": round(predicted_price, 2),
        "message": "Prediction successful"
    }
