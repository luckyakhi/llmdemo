import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Generate Sample Dataset (Symptoms -> Disease)
np.random.seed(42)

# Simulated symptoms dataset (10 symptoms, binary presence: 0 or 1)
num_samples = 1000
num_symptoms = 10

X = np.random.randint(0, 2, size=(num_samples, num_symptoms))  # Symptom presence (0 or 1)
y = np.random.choice(['Flu', 'COVID-19', 'Malaria', 'Dengue'], num_samples)  # Disease labels

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardizing input features (optional but improves training performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Build Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(num_symptoms,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer with softmax
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Step 4: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Step 5: Make Predictions (Example Input)
new_symptoms = np.random.randint(0, 2, (1, num_symptoms))  # Random new patient symptoms
new_symptoms_scaled = scaler.transform(new_symptoms)  # Apply same scaling
prediction = model.predict(new_symptoms)
predicted_disease = label_encoder.inverse_transform([np.argmax(prediction)])

print(f"Predicted Disease: {predicted_disease[0]}")
