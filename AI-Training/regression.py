import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('weather_data.csv',low_memory=False)
data = data.sample(frac=1, random_state=2)
data.reset_index(drop=True, inplace=True)
print(data.shape)
print(data.info())
dropable = 'rhum', 'msl', 'wdsp', 'wddir', 'ww', 'w', 'sun'
data.drop(columns=dropable, inplace=True)

y = data['temp']
data = data.replace('', float('nan'))
data = data.dropna()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1) 
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print("Mean Squared Error:", loss)