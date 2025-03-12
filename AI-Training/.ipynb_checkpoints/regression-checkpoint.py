import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

RAIN_THRESHHOLD = 0.5
# Load data
data = pd.read_csv('weather_data.csv', low_memory=False)
data = data.sample(frac=1, random_state=2).reset_index(drop=True)

print("Data shape:", data.shape)

# Preprocessing
# Handle missing values
data = data.replace('', float('nan')).dropna()

# Separate features and target, convert to encoding
y = data['rain'].apply(lambda x: 1 if x > RAIN_THRESHHOLD else 0)

X = data.drop(columns=['rain', 'date'])

# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Ensure all data is numeric
X = X.astype(float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure target variable is integer-encoded for sparse_categorical_crossentropy
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Build the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer
    layers.Dense(64, activation='relu'),        # Hidden layer 1
    layers.Dense(32, activation='relu'),        # Hidden layer 2
    layers.Dense(2, activation='softmax')       # Output layer (2 classes)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

sample_input_1 = np.array([[0,0,9.2,0,9.2,9.1,11.6,100,992.1,2,23,2,160,65,66,0.0,2000,4,8]])
sample_input_2 = np.array([[0,0,3.8,0,2.9,1.6,6.8,85,1020.8,2,5,2,80,2,11,0.0,30000,999,1]])

pred_1 = model.predict(sample_input_1)
pred_2 = model.predict(sample_input_2)
# Interpret predictions
pred_1_class = np.argmax(pred_1, axis=1)
pred_2_class = np.argmax(pred_2, axis=1)

print("Prediction for sample 1:", pred_1_class)
print("Prediction for sample 2, should be 0:", pred_2_class)
