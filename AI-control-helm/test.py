from random import randint
from time import sleep
from tkinter import E
from typing import List
from flask import Flask, request, jsonify
import threading
import requests
import os
import numpy as np
import tensorflow as tf

requests.packages.urllib3.util.connection.HAS_IPV6 = False

CLUSTER_NAME =os.getenv("CLUSTER_NAME")
CLUSTER_RANK =os.getenv("CLUSTER_RANK")
PORT =os.getenv("PORT")

INTERVAL = 3 # Maybe env var
COLUMNS = ["ind","indt","temp","indw","wetb","dewpt","vappr","rhum","msl","indm","wdsp","indwd","wddir","ww","w","sun","vis","clht","clamt"] # env var 
BUFFER_SIZE = 10 # Env var 
RAIN_THRESHOLD = 0.5
MODEL_PATH = "/app/model/model.tflite"
PROFILE_URL = f"http://{CLUSTER_NAME}.local/core-metadata/api/v3/device/profile/name/Generic-MQTT-String-Float-Device"
CONTROL_URL = "http://control.local"
CONTROL_AI_URL = CONTROL_URL+f"/{CLUSTER_NAME}"

loaded_model = tf.keras.models.load_model(MODEL_PATH)

data_lock = threading.Lock()
# Incoming data added to buffer and is reset on interval
incoming_data = {}
listeners = []
data_buffer = []
clusters = {}
sensors ={}
app = Flask(__name__)

@app.route('/edge_result', methods=['POST'])
def edge_result():
    json_data = request.json 
    if not json_data:
        return jsonify({"error": "No data provided"}), 400
    nodes = json_data.get('node_data')
    result = json_data.get('result')
    inference(nodes,result)

## Vars that need to be defined for the function 
# Recieve node_data, data_result
BATCH_SIZE = 5
X_data = []  
Y_data = []  

def inference(node_data,data_result):
    node_data = np.array(node_data)
    data_result = np.array([data_result])
    edge_model_result = node_data
    edge_model_result = np.reshape(edge_model_result, (-1, 32)) 
    control_predictions = loaded_model.predict(edge_model_result)
    print("Control predictions:", control_predictions)

    #IF data result exists, add it and intermediate layer to batches
    try:
        if data_result.size > 0: 
        #Add data_result to Y and beginnging nodes to X
            X_data.append(edge_model_result) 
            Y_data.append(data_result) 
            if len(X_data) >= BATCH_SIZE:
                numpy_X_data = np.vstack(X_data)
                numpy_Y_data = np.vstack(Y_data)  

                # Clear lists for the next batch
                X_data.clear()
                Y_data.clear()

                # Compile and train the control_model
                loaded_model.compile(optimizer='adam',
                                    loss='mean_squared_error',
                                    metrics=['mae'])
                print(f"Training Model with batch size:{BATCH_SIZE}")
                history = loaded_model.fit(numpy_X_data, numpy_Y_data,
                                            epochs=1, batch_size=BATCH_SIZE)
    except:
        print("An error occured in training")

while True:
    app.run(host='0.0.0.0', port=PORT)
