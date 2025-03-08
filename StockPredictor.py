import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Generate Synthetic Data
# -----------------------------
# For demonstration, assume we have 3 financial features:
# - revenue (in millions)
# - profit_margin (a fraction)
# - debt_ratio (a fraction)

np.random.seed(42)  # for reproducibility
n_samples = 100

# Generate random financial data for 100 companies
revenue = np.random.uniform(100, 1000, n_samples)      # revenue between 100 and 1000
profit_margin = np.random.uniform(0, 0.3, n_samples)     # profit margin between 0 and 30%
debt_ratio = np.random.uniform(0, 1, n_samples)          # debt ratio between 0 and 1

# Combine features into a single input matrix X
X = np.column_stack([revenue, profit_margin, debt_ratio])

# Define a simple relationship for the stock price (target variable y)
# For example, a basic formula might be:
# stock price = 0.1 * revenue + 200 * profit_margin - 50 * debt_ratio + some noise
noise = np.random.normal(0, 10, n_samples)
y = 0.1 * revenue + 200 * profit_margin - 50 * debt_ratio + noise

# ------------------------------------
# 2. Data Preprocessing (Normalization)
# ------------------------------------
# Neural networks perform better with normalized data.
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# ------------------------------------
# 3. Define a Minimal Neural Network
# ------------------------------------
# Here, we build a simple model with:
# - An input layer (implicitly defined by input_dim)
# - One hidden layer with 3 nodes (keeps the network minimal)
# - An output layer with 1 node (predicting a continuous stock price)

model = Sequential()
model.add(Dense(3, input_dim=3, activation='relu'))  # Hidden layer with 3 nodes
model.add(Dense(1))  # Output layer

# Compile the model with mean squared error loss and the Adam optimizer
model.compile(loss='mse', optimizer='adam')

# -----------------------------
# 4. Train the Neural Network
# -----------------------------
# Train for 100 epochs; verbose=0 suppresses the training output
model.fit(X_scaled, y_scaled, epochs=100, verbose=0)

# ------------------------------------
# 5. Making a Prediction
# ------------------------------------
# For demonstration, predict the stock price for the first sample from our data.
predicted_scaled = model.predict(X_scaled[0:1])
# Convert the normalized prediction back to the original scale
predicted_price = scaler_y.inverse_transform(predicted_scaled)

print("Predicted stock price for the first sample:", predicted_price[0][0])
