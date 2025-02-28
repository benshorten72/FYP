from glob import glob
from time import sleep
from flask import Flask, request, jsonify
import threading
import requests
import os
import numpy as np
import tensorflow as tf
import json
from datetime import datetime

requests.packages.urllib3.util.connection.HAS_IPV6 = False
import logging

# Disable Flask/Werkzeug logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CLUSTER_NAME = os.getenv("CLUSTER_NAME")
CLUSTER_RANK = os.getenv("CLUSTER_RANK")
PORT = os.getenv("PORT")
INTERVAL = 3  # Maybe env var
COLUMNS = ["ind", "indt", "temp", "indw", "wetb", "dewpt", "vappr", "rhum", "msl", "indm", "wdsp", "indwd", "wddir", "ww", "w", "sun", "vis", "clht", "clamt"]  # env var
BUFFER_SIZE = 2  # Env var
RAIN_THRESHOLD = 0.5
MODEL_PATH = "/app/model/model.keras"
PROFILE_URL = f"http://{CLUSTER_NAME}.local/core-metadata/api/v3/device/profile/name/Generic-MQTT-String-Float-Device"
CONTROL_URL = "http://control.local"
CONTROL_AI_URL = CONTROL_URL + f"/{CLUSTER_NAME}"
MAX_DATASET_SIZE = 1000  
BATCH_SIZE = 16

loaded_model = tf.keras.models.load_model(MODEL_PATH)
loaded_model.compile(optimizer='adam',
                     loss='mean_squared_error',
                     metrics=['mae'])
data_lock = threading.Lock()
# Incoming data added to buffer and is reset on interval
data_buffer = []
app = Flask(__name__)
app.logger.disabled = True
numpy_X_data = np.empty((0, 32))  # Adjust shape as needed
numpy_Y_data = np.empty((0, 1))
## Vars that need to be defined for the function
# Recieve node_data, data_result
weights = None
X_data = []
Y_data = []
fed_weights = None
print("Port:",PORT)

@app.route('/edge_result', methods=['POST'])
def edge_result():
    try:

        json_data = request.get_json(force=True)
        if json_data is None:
            print("Failed to parse JSON data")
            return jsonify({"error": "Invalid JSON data"}), 400

        # Validate required fields in the JSON data
        if "node_data" not in json_data or "name" not in json_data:
            return jsonify({"error": "Missing required fields in JSON data"}), 400

        node_data = json_data["node_data"]
        name = json_data["name"]
        result = json_data.get("result")

        print(f"Processing data for node: {name}")
        with data_lock:
            data_buffer.append({"node_data": node_data, "result": result})
        return jsonify({"message": "Data received successfully", "name": name, "data_length": len(node_data)}), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500
    

def ai_thread():
    global weights, fed_weights
    while True:
        data_point = None
        #Update to fed weights if different to current weights, I have it here awkwardly
        # Because I dont want weights updated when inference or training is occuring
        if fed_weights and fed_weights != weights:
            weights = fed_weights
            loaded_model.set_weights(weights)

        with data_lock:
            if data_buffer:
                data_point = data_buffer.pop(0)
            
        if data_point:
            print("Beginning Inference", flush=True)
            inference(data_point["node_data"], data_point["result"])
        sleep(1)

def inference(node_data, data_result):
    global numpy_X_data, numpy_Y_data, weights
    node_data = np.array(node_data)
    data_result = np.array([data_result])
    edge_model_result = node_data
    edge_model_result = np.reshape(edge_model_result, (-1, 32))
    control_predictions = loaded_model.predict(edge_model_result)
    print("Control predictions:", control_predictions, flush=True)
    # If result does not exist do not do training
    if data_result == "empty" or data_result == 'empty':
        print("No result", flush=True)
        return
    # IF data result exists, add it and intermediate layer to batches
        # Add data_result to Y and beginnging nodes to X
    X_data.append(edge_model_result)
    Y_data.append(data_result)
    print(f"Buffer:{len(X_data)}/{BUFFER_SIZE}")

    if len(X_data) >= BUFFER_SIZE:
        for i in range(len(X_data)):
           numpy_X_data = np.concatenate((numpy_X_data, X_data[i].reshape(1, -1)), axis=0)
           numpy_Y_data = np.concatenate((numpy_Y_data, Y_data[i].reshape(1, -1)), axis=0)

        # Clear lists for the next batch
        X_data.clear()
        Y_data.clear()

        # Compile and train the control_model
        print(f"Training Model with batch size: {BATCH_SIZE}", flush=True)
        history = loaded_model.fit(numpy_X_data, numpy_Y_data,
                                    epochs=5, batch_size=BATCH_SIZE)

        if len(numpy_X_data) > MAX_DATASET_SIZE:
            numpy_X_data = numpy_X_data[-MAX_DATASET_SIZE:]
            numpy_Y_data = numpy_Y_data[-MAX_DATASET_SIZE:]

        weights = loaded_model.get_weights()

@app.route('/get_weights', methods=['GET'])
def get_weights():
    try:
        weights_json = [w.tolist() for w in weights]
        return jsonify({"weights": weights_json}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_weights', methods=['POST'])
def set_weights():
    global fed_weights
    json_data = request.get_json(force=True)
    json_weights = json_data["weights"]
    fed_weights = [np.array(w) for w in json_weights]
    # Serialize to JSON
    return jsonify({"Scuess": 'good'}), 200

def run_flask_app():
    app.run(host='0.0.0.0', port=PORT)


# Start the Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.daemon = True
flask_thread.start()

# Start the AI thread
thread_inference = threading.Thread(target=ai_thread)
thread_inference.daemon = True
thread_inference.start()

# Keep the main thread alive to keep the daemon threads running
while True:
    sleep(1)