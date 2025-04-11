from glob import glob
from time import sleep
from flask import Flask, request, jsonify
import threading
import requests
import os
import numpy as np
import tensorflow as tf
import json
import psutil
from datetime import datetime
from time import time

requests.packages.urllib3.util.connection.HAS_IPV6 = False
import logging

# Disable Flask/Werkzeug logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CLUSTER_NAME = os.getenv("CLUSTER_NAME")
CLUSTER_RANK = os.getenv("CLUSTER_RANK")
SPLIT_LEARNING = os.getenv("SPLIT_CHECK")
USE_EARLY_CHECK = os.getenv("USE_EARLY_STOPPING")

if SPLIT_LEARNING.lower()=="true":
    temp=True
else:
    temp=False
SPLIT_CHECK=temp

if USE_EARLY_CHECK.lower()=="true":
    temp=True
else:
    temp=False
USE_EARLY_STOPPING=temp
print(f"Split Learning enabled: {SPLIT_CHECK}")

PORT = os.getenv("PORT")
INTERVAL = 3  # Maybe env var
BUFFER_SIZE = 10  # Env var
RAIN_THRESHOLD = 0.5
MODEL_PATH = "/app/model/model.keras"
PROFILE_URL = f"http://{CLUSTER_NAME}.local/core-metadata/api/v3/device/profile/name/Generic-MQTT-String-Float-Device"
CONTROL_URL = "http://control.local"
CONTROL_AI_URL = CONTROL_URL + f"/{CLUSTER_NAME}"
METRICS_SERVER= "http://control.local/metrics/add_metrics"
LEARNING_THRESHOLD= 200
MAX_DATASET_SIZE = 10000
BATCH_SIZE = 16
VAL_SPLIT_RATIO = 0.2  # 20% of BUFFER_SIZE for validation

best_val_loss = float('inf')
patience = 3 
wait = 0
loaded_model = tf.keras.models.load_model(MODEL_PATH)
loaded_model.compile(optimizer='adam',
                     loss='mean_squared_error',
                     metrics=['mae'])
data_lock = threading.Lock()
# Incoming data added to buffer and is reset on interval
data_buffer = []
app = Flask(__name__)
app.logger.disabled = True

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
print(f"Max content length: {app.config['MAX_CONTENT_LENGTH']} bytes")
    
numpy_X_data = np.empty((0, 32))
numpy_Y_data = np.empty((0, 1))
numpy_X_train = np.empty((0, 32))
numpy_Y_train = np.empty((0, 1))
numpy_X_val = np.empty((0, 32))
numpy_Y_val = np.empty((0, 1))

## Vars that need to be defined for the function
# Recieve node_data, data_result
weights = loaded_model.get_weights() 
X_data = []
Y_data = []
fed_weights = None
print("Port:",PORT)

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

def send_metrics(data_name,values):
    try:    
        response=requests.post(METRICS_SERVER,json={'cluster_name':CLUSTER_NAME,'data_name':data_name,'time':datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        'values':values})
        response.raise_for_status() 
        print(f"Metrics sent successfully: {data_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send metrics: {e}")
    except ValueError as e:
        print(f"Invalid data format: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")   
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
        result = float(json_data.get("result"))
        raw = json_data.get("raw")
        if raw is not None:
            print("Raw data present for back propagation")
        else:
            raw = None  

        print("Actual data result:",result)
        print(f"Processing data for node: {name}")
        with data_lock:
            data_buffer.append({"node_data": node_data, "result": result,"raw":raw})
        return jsonify({"message": "Data received successfully", "name": name, "data_length": len(node_data)}), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500
    

def ai_thread():
    global weights, fed_weights
    while True:
        try:
            data_point = None
            if fed_weights is not None and not all(np.array_equal(fed_w, w) for fed_w, w in zip(fed_weights, weights)):
                print("Federated update incoming")
                weights = fed_weights
                loaded_model.set_weights(weights)
                print("Weights updated")

            with data_lock:
                if data_buffer:
                    data_point = data_buffer.pop(0)

            if data_point:
                start_time = time()
                inference(data_point["node_data"], data_point["result"],data_point["raw"])
                inference_time = time() - start_time
                send_metrics("control_inference_time", [inference_time])
            sleep(1)
        except Exception as e:
            print(f"AI Thread Error: {e}", flush=True)

def inference(node_data, data_result,raw):
    global numpy_X_data, numpy_Y_data, numpy_X_train, numpy_Y_train, numpy_X_val, numpy_Y_val,weights, best_val_loss, wait
    node_data = np.array(node_data)
    print(type(data_result))
    if data_result!= 0.0:
        data_result = data_result/1000

    data_result = np.array([data_result])
    edge_model_result = node_data
    edge_model_result = np.reshape(edge_model_result, (-1, 32))
    control_predictions = loaded_model.predict(edge_model_result)
    send_metrics("control_inference_result", [float(abs(control_predictions.tolist()[0][0]))])
    send_metrics("control_inference_vs_result", [abs(float(abs(control_predictions.tolist()[0][0]))-data_result[0])])
    print("Inference Result:",abs(control_predictions.tolist()[0][0]),flush=True)
    # If result does not exist do not do training
    if data_result == "empty" or data_result == 'empty':
        print("No result", flush=True)
        return
    else: 
        print(data_result,flush=True)
    # IF data result exists, add it and intermediate layer to batches
        # Add data_result to Y and beginnging nodes to X
    X_data.append(edge_model_result)
    Y_data.append(data_result)

    print(f"Buffer:{len(X_data)}/{BUFFER_SIZE}")

    if SPLIT_CHECK:
        if raw == None:
            print("Missing original data input to send back to control to complete back propagation")
        else:
            edge_model_result = tf.convert_to_tensor(edge_model_result, dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape: # Remember this is not training the data, we are simply getting gradients
                tape.watch(edge_model_result)
                control_predictions = loaded_model(edge_model_result, training=True)
                loss = tf.keras.losses.MeanSquaredError()(data_result, control_predictions)

            gradients = tape.gradient(loss, edge_model_result)
            gradients_serializable = [grad.numpy().tolist() for grad in gradients]
            
            # Send the cut gradient to the head model via HTTP
            payload = {"gradients": gradients_serializable,"raw":raw,"result":data_result.tolist()}

            back_prop_url=f"http://{CLUSTER_NAME}.local/ai/back_propagate"
            print(f"Backpropagating weights to edge on",back_prop_url,flush=True)

            response = requests.post(back_prop_url, json=payload)
            send_metrics("control_gradient_update_sent", [1])
            print(response,flush=True)
            
    if len(X_data) >= BUFFER_SIZE:
        # Split into training/validation sets
        split_idx = int(BUFFER_SIZE * (1 - VAL_SPLIT_RATIO))
        X_train, X_val = X_data[:split_idx], X_data[split_idx:]
        Y_train, Y_val = Y_data[:split_idx], Y_data[split_idx:]
        
        for i in range(len(X_data)):
            numpy_X_data = np.concatenate((numpy_X_data, X_data[i].reshape(1, -1)), axis=0)
            numpy_Y_data = np.concatenate((numpy_Y_data, Y_data[i].reshape(1, -1)), axis=0)
        for i in range(len(X_train)):
            numpy_X_train = np.concatenate((numpy_X_train, X_train[i].reshape(1, -1)), axis=0)
            numpy_Y_train = np.concatenate((numpy_Y_train, Y_train[i].reshape(1, -1)), axis=0)
        for i in range(len(X_val)):
            numpy_X_val = np.concatenate((numpy_X_val, X_val[i].reshape(1, -1)), axis=0)
            numpy_Y_val = np.concatenate((numpy_Y_val, Y_val[i].reshape(1, -1)), axis=0)

        # Clear lists for the next batch
        X_data.clear()
        Y_data.clear()
        if len(numpy_X_data) <= LEARNING_THRESHOLD:
            print(f"Dataset not large enough to preform learning {len(numpy_X_data)}/{LEARNING_THRESHOLD}")
        else:
        # Compile and train the control_model
            print(f"Training Model with batch size: {BATCH_SIZE}", flush=True)
            print(f"Total values in storage:",len(numpy_X_train))
            start_time = time()

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            train_dataset = tf.data.Dataset.from_tensor_slices((numpy_X_train, numpy_Y_train))
            train_dataset = train_dataset.batch(BATCH_SIZE)

            # Validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((numpy_X_val, numpy_Y_val))
            val_dataset = val_dataset.batch(BATCH_SIZE)

            for epoch in range(2):
                # Training loop
                for batch_X, batch_Y in train_dataset:
                    with tf.GradientTape() as tape:
                        predictions = loaded_model(batch_X, training=True)
                        train_loss = tf.keras.losses.MeanSquaredError()(batch_Y, predictions)
                    gradients = tape.gradient(train_loss, loaded_model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, loaded_model.trainable_variables))

                val_loss = 0
                for val_X, val_Y in val_dataset:
                    val_pred = loaded_model(val_X, training=False)
                    val_loss += tf.keras.losses.MeanSquaredError()(val_Y, val_pred)
                val_loss /= len(val_dataset)

                # Early stopping logic
                if USE_EARLY_STOPPING:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"Early stopping at epoch {epoch+1}!",flush=True)
                            send_metrics("early_learning_epoch_stop",[epoch+1])
                            break
                print(f"Epoch {epoch+1}, Train Loss: {train_loss.numpy()}, Val Loss: {val_loss.numpy()}")
                print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
            training_time = time() - start_time
            send_metrics("control_training_loss", [float(loss.numpy())])
            send_metrics("control_training_time", [training_time])
            send_metrics("dataset_size", [len(numpy_X_data)])

            if len(numpy_X_data) > MAX_DATASET_SIZE:  # Trims if needed
                numpy_X_data = numpy_X_data[-MAX_DATASET_SIZE:]
                numpy_Y_data = numpy_Y_data[-MAX_DATASET_SIZE:]

            weights = loaded_model.get_weights()



@app.route('/get_weights', methods=['GET'])
def get_weights():
    try:
        weights_json = [w.tolist() for w in weights]
        return jsonify({"weights": weights_json,"data_amount":len(numpy_X_data)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_weights', methods=['POST'])
def set_weights():
    global fed_weights
    try:
        print("Weights federating incoming")
        json_data = request.get_json(force=True)
        
        if not json_data or "weights" not in json_data:
            return jsonify({"error": "Invalid input: missing weights"}), 400
            
        # Convert and store weights
        json_weights = json_data["weights"]
        fed_weights = [np.array(w, dtype=np.float64) for w in json_weights]
        current_weights = loaded_model.get_weights()
        
        if len(current_weights) != len(fed_weights):
            return jsonify({"error": "Weight shape mismatch"}), 400
            
        diffs = []
        for i in range(len(current_weights)):
            # Safe division with epsilon to avoid divide-by-zero
            denominator = np.where(
                np.abs(fed_weights[i]) < 1e-10, 
                1e-10,
                fed_weights[i]
            )
            relative_diff = np.abs(current_weights[i] - fed_weights[i]) / denominator
            diffs.append(relative_diff)
        
        sums = np.array([np.nansum(x) for x in diffs])
        counts = np.array([np.count_nonzero(~np.isnan(x) & ~np.isinf(x)) for x in diffs])
        
        if np.sum(counts) == 0:
            overall_change = 0  
        else:
            overall_change = np.sum(sums) / np.sum(counts)
        
        # Log metrics
        send_metrics("Weight difference after federation", [float(overall_change)])
        
        return jsonify({"success": True, "avg_change": float(overall_change)}), 200
        
    except Exception as e:
        print(f"Error in set_weights: {str(e)}")
        return jsonify({"error": str(e)}), 500

def run_flask_app():
    app.run(host='0.0.0.0', port=PORT,threaded=True)


# Start the Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.daemon = True
sleep(2)
flask_thread.start()

# Start the AI thread
thread_inference = threading.Thread(target=ai_thread)
thread_inference.daemon = True
thread_inference.start()

def watchdog_thread(target_thread):
    while True:
        if not target_thread.is_alive():
            print("AI thread is not alive!", flush=True)
        else:
            print("AI thread is running", flush=True)
        sleep(10)

watchdog = threading.Thread(target=watchdog_thread, args=(thread_inference,))
watchdog.daemon = True
watchdog.start()

# Keep the main thread alive to keep the daemon threads running
print("Gettting federated weights")
requests.post("https://control.local/parameter-server/update_my_weights",{"name":CLUSTER_NAME},verify=False)

while True:
    send_metrics("control_python_memory_usage",[get_process_memory()])
    sleep(2)