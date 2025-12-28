# House Price Prediction using Machine Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
# Sample housing dataset (you can replace with CSV later)
data = {
    'Area': [800, 900, 1200, 1500, 1800, 2000, 2200, 2500],
    'Bedrooms': [1, 2, 2, 3, 3, 4, 4, 5],
    'Bathrooms': [1, 1, 2, 2, 3, 3, 4, 4],
    'Price': [50000, 60000, 90000, 120000, 150000, 180000, 210000, 250000]
}

df = pd.DataFrame(data)

# -----------------------------
# Data Preprocessing
# -----------------------------
X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Linear Regression Model
# -----------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

# -----------------------------
# Random Forest Model
# -----------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------
print("Linear Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\nRandom Forest:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# -----------------------------
# Prediction Example
# -----------------------------
# New house: Area=1600 sqft, Bedrooms=3, Bathrooms=2
new_house = np.array([[1600, 3, 2]])
predicted_price = rf_model.predict(new_house)

print("\nPredicted Price for new house:", int(predicted_price[0]))
