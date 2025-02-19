import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('weather_data.csv', low_memory=False)
data = data.sample(frac=1, random_state=2).reset_index(drop=True)

print(data.shape)
print(data.info())
#Get min measurement 
y = data['sun']
X = data.drop(columns=['sun']) 

for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

X = X.replace('', float('nan')).dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 # classification - use softmax
X_train = X_train.astype(float)
X_test = X_test.astype(float)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(loss)

# Temp loss: 1.134574055671692 *pretty good*
# Rain loss: 0.5148706436157227 ?
# Sun: 0.10442148894071579

