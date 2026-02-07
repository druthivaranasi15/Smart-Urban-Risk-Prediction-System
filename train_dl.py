import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("--- Starting DEEP LEARNING Training ---")

# 1. LOAD DATA
df = pd.read_csv('data/processed_data.csv')
X = df.drop(columns=['Target'])
y = df['Target']

# 2. SCALE DATA (Crucial for Neural Networks)
# Deep Learning fails if numbers are big (like "2023"). We squeeze them between 0 and 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler so the App can use it later
joblib.dump(scaler, 'data/scaler.pkl')

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. CALCULATE CLASS WEIGHTS (To fix imbalance naturally)
# This tells the brain: "Pay 3x more attention to Fatal accidents"
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weight_dict}")

# 5. BUILD THE NEURAL NETWORK
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]), # Input Layer
    layers.Dropout(0.3), # Prevents overfitting (forgetting random things)
    layers.Dense(32, activation='relu'), # Hidden Layer
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid') # Output Layer (0 to 1 probability)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. TRAIN THE BRAIN
print("Training Neural Network...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=50, # Go through the data 50 times
    class_weight=class_weight_dict,
    verbose=1
)

# 7. EVALUATE
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n>>> DEEP LEARNING ACCURACY: {accuracy * 100:.2f}%")

# 8. SAVE
model.save('data/deep_model.keras')
print("Model saved to 'data/deep_model.keras'")