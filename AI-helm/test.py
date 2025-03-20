from random import randint
from time import sleep
from typing import List
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify
import threading
import requests
import os
import numpy as np
import json
from datetime import datetime

from ai_edge_litert.interpreter  import Interpreter
requests.packages.urllib3.util.connection.HAS_IPV6 = False
import os
import sys
import stat
CLUSTER_NAME =os.getenv("CLUSTER_NAME")
CLUSTER_RANK =os.getenv("CLUSTER_RANK")
SPLIT_LEARNING=os.getenv("SPLIT_CHECK")

INTERVAL = 3 # Maybe env var
COLUMNS = ["ind","indt","temp","indw","wetb","dewpt","vappr","rhum","msl","indm","wdsp","indwd","wddir","ww","w","sun","vis","clht","clamt","result"] # env var 
BUFFER_SIZE = 10 # Env var 
RAIN_THRESHOLD = 0.5
PROFILE_URL = f"http://{CLUSTER_NAME}.local/core-metadata/api/v3/device/profile/name/Generic-MQTT-String-Float-Device"
CONTROL_URL = "http://control.local"
CONTROL_AI_URL = CONTROL_URL+f"/{CLUSTER_NAME}"
METRICS_SERVER= "http://control.local/metrics/add_metrics"
data_lock = threading.Lock()
training_lock = threading.Lock()
# Incoming data added to buffer and is reset on interval
incoming_data = {}
listeners = []
data_buffer = []
client = mqtt.Client()
clusters = {}
sensors ={}
app = Flask(__name__)
print("Split learning enabled:",SPLIT_LEARNING)

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

if SPLIT_LEARNING.lower()=="true":
    temp=True
    MODEL_PATH = "/app/model/model.keras"
    from inferenceHeavy import inference_data


    import tensorflow as tf
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    loaded_model.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['mae', 'mae'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    for layer in loaded_model.layers:
        layer.trainable = True
    loaded_model.get_layer("edge_output").trainable = False
else:
    temp=False
    from inferenceLite import inference_data
    MODEL_PATH = "/app/model/model.tflite"
    # AI
    loaded_model = Interpreter(model_path=MODEL_PATH)
    signatures = loaded_model.get_signature_list()
    loaded_model.allocate_tensors()

    # GET THE OUTPUT DATA
    loaded_model.invoke()
    output_details = loaded_model.get_output_details()
    output_data = loaded_model.get_tensor(output_details[0]['index'])
    
SPLIT_CHECK=temp

for i in COLUMNS:
    incoming_data[i]=None
def get_clusters():
    try:
        response = requests.get(CONTROL_URL + "/control/get_clusters")
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return

    if "clusters" in data:
        clusters.clear()
        # Iterate over the dictionary items (key-value pairs)
        for name, rank in data["clusters"].items():
            if name != CLUSTER_NAME:
                clusters[name] = rank


def mqtt_thread():
    def on_message(client, userdata, message):
        # print(f"Received data on topic: {message.topic}")
        for listener in listeners:
            if listener.topic == message.topic:
                with data_lock:
                    listener.handle_message(client, userdata, message)
                break

    client.on_message = on_message
    client.connect("edgex-mqtt-broker", 1883, 60)
    client.loop_start()
    
class Listener:
    def __init__(self, topic, name):
        self.topic = topic
        self.name = name
        client.subscribe(topic)
    
    def handle_message(self, client, userdata, message):
        data = message.payload.decode()
        incoming_data[self.name] = data
        # print(f"Received from {self.name}: {data}")

    def unsubscribe(self):
        client.unsubscribe(self.topic)
# I need to now
# 1. Get all devices periodcally
# 2. Create new mqtt subscriber for devices if it doesnt exist
# 3. Create a buffer to recieve and compile data within interval into example
def fetch_devices():
    try:
        #Fetch clusters
        get_clusters()
        print(f"Other clusters:{clusters}")
        print(f"Fetching devices on{PROFILE_URL}")
        response = requests.get(PROFILE_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        devices = response.json()
        # Check if "devices" key exists and is not empty
        if "devices" not in devices or not devices["devices"]:
            print(f"No devices detected in EdgeX MQTT service: {PROFILE_URL}")
            return
        devices_present = []
        for device in devices["devices"]:
            device_name = device.get("name")
            profile_name = device.get("profileName")
            topic = device.get("protocols", {}).get("mqtt", {}).get("topic")
            devices_present.append(device_name)
            # Check if required fields are present
            if not all([device_name, profile_name, topic]):
                print(f"Skipping device due to missing fields: {device}")
                continue

            print(f" - Device Name: {device_name}")
            print(f" - Profile Name: {profile_name}")
            print(f" - MQTT Topic: {topic}\n")
            # Check if device_name is in COLUMNS and not already in sensors
            if device_name in COLUMNS: 
                if device_name not in sensors:
                    print(f"Adding new listener for device: {device_name}")
                    listen = Listener(topic, device_name)
                    listeners.append(listen)
                    sensors[device_name] = listen
            else:
                print(f"{device_name} is not apart of list of predefined data names/columns: {COLUMNS}")

        # Handle incoming data after processing devices


        # Print columns that are not being listened for
        unlistened_columns = [col for col in COLUMNS if col not in sensors]
        if unlistened_columns:
            print(f"Columns not being listened for: {unlistened_columns}")
        else:
            print("All columns are being listened for.")

        handle_incoming_data()
        print(f"Buffer has been updated to contain: {data_buffer}")
        # Unsubscribe from devices not associated with device profile
        if len(sensors.keys()) > 0:
            for sensor_name in list(sensors.keys()):
                if sensor_name not in devices_present:
                    for listener in sensors[sensor_name]:
                        listener.unsubscribe()
                        listeners.remove(listener) 
                    del sensors[sensor_name]
                    print(f"Removed sensor: {sensor_name} as is no longer associatd with device profile")
        print("-" * 40)
        print("\n"*3)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching devices: {e}")

def handle_incoming_data():
    global data_buffer
    # Check if number of connected devices is equal to COLUMNS and that buffer size isnt overflowing
    print(f"Current listeners: {len(listeners)}")
    # Check if all the listners needed are present EXCEPT result
    len_listeners = len(listeners)
    len_columns = len(COLUMNS)
    if "result" in COLUMNS:
        len_columns-=1
    for i in listeners:
        if i.name == "result":
            len_listeners -=1
            break
    # Process data if correct amount of listeners are present
    if len_listeners==len_columns:
        if len(data_buffer) >= BUFFER_SIZE:
            data_buffer.pop(0)

        print(f"Current incoming data: {incoming_data}")
        with data_lock:
            processed_data = dict(incoming_data)
            processed_data["rank"] = CLUSTER_RANK
            processed_data["cluster_name"] = CLUSTER_NAME
            # Add result = None, if no result has been added
            # This means that on the model it will preform only inference
            # If result does exist it will use it as labled data
            if not "result" in processed_data.keys():
                processed_data["result"]=None
            data_buffer.append(dict(processed_data))
            #Reset incoming data after interval 
            for column in incoming_data.keys():
                incoming_data[column] = None
    send_metrics("edge_data_buffer_size",[len(data_buffer)])
    if len(data_buffer) == BUFFER_SIZE:
        buffers = get_buffers()
        if not buffers:
            print("No buffersavalible to distribute load")
        else:
            print("Buffers avalible to distribute load")
            partioned_data = data_buffer[len(data_buffer)//2:]
            with data_lock:
                print("Data buffer halved")
                data_buffer = data_buffer[:len(data_buffer)//2]
            make_space(partioned_data,buffers)

thread = threading.Thread(target=mqtt_thread)
thread.daemon = True 
thread.start()
# ---------------------------------------
@app.route('/back_propagate',methods=['POST'])
def back_propagate():
    data = request.get_json()
    print("Back propagation received. Applying to model", flush=True)

    for layer in loaded_model.layers:
        layer.trainable = True
    loaded_model.get_layer("edge_output").trainable=False

    if SPLIT_CHECK:
        for layer in loaded_model.layers:
            layer.trainable = True
        loaded_model.get_layer("edge_output").trainable = False
        received_gradients = [tf.reshape(tf.convert_to_tensor(data["gradients"], dtype=tf.float32), (1, -1))]        
        original_input_data = np.array([data["raw"]],dtype=np.float32)

        with training_lock:
            with tf.GradientTape() as tape:
                edge_output = loaded_model(original_input_data, training=True)
            tape.watch(edge_output)
            edge_model_grads = tape.gradient(edge_output, loaded_model.trainable_variables, output_gradients=received_gradients)
            optimizer.apply_gradients(zip(edge_model_grads, loaded_model.trainable_variables))

        for layer in loaded_model.layers:
                layer.trainable = False
        loaded_model.get_layer("edge_output").trainable = True
        temp_result = np.array([data["result"]])
        temp_result = temp_result.reshape(-1, 1) 
        dummy_output = np.zeros((1, 32), dtype=np.float32)

        temp_result = np.array(data["result"], dtype=np.float32).reshape(1, 1)

        final_output = [dummy_output, temp_result]
        print("Original input shape:", original_input_data.shape)
        print("First output shape:", dummy_output.shape)
        print("Second output shape:", temp_result.shape)
        print("Final output type:", type(final_output))
        print("Final output length:", len(final_output),flush=True)
        # Train the model
        loaded_model.fit(original_input_data, final_output, epochs=1, batch_size=1)
        
        print("Applied to model", flush=True)
        return jsonify({"message": "Gradients applied successfully!"})

    return jsonify({"message": "Gradients cannot be applied due to split learning disabled"})

@app.route('/get_buffer', methods=['GET'])
def get_buffer():
    return jsonify({"buffer": data_buffer}), 200

@app.route('/fit_data', methods=['POST'])
def fit_data():
    global data_buffer
    try:
        data_to_be_fit = request.json 
        if not data_to_be_fit:
            return jsonify({"error": "No data provided"}), 400
        with data_lock:
            print("\n",flush=True)
            print("*"*40,flush=True)
            print("--- Recieving incoming data to fit ---",flush=True)
            data_buffer.extend(data_to_be_fit)            
            sorted_data = sorted(data_buffer, key=lambda x: x["rank"])
            data_buffer = sorted_data[:10]
            print(f"Received buffer",flush=True)
            print(f"Loosing data:",len(sorted_data[10:]),flush=True)
            print("*"*40,flush=True)
            print("\n",flush=True)

        return jsonify({"message": "Data received and processed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=5000)

thread = threading.Thread(target=run_flask)
thread.daemon = True 
thread.start()

def get_buffers():
    buffers={}
    if clusters:
        for cluster in clusters:
            if int(clusters[cluster]) > int(CLUSTER_RANK):
                print(f"\n requesting buffer from: http://{cluster}.local/ai/get_buffer ")
                response = requests.get(f"http://{cluster}.local/ai/get_buffer")
                response.raise_for_status() 
                data = response.json()
                buffers[cluster] = data.get("buffer")
                print("Data retrived:",buffers[cluster],"\n")
        return buffers
    else:
        print("No other clusters avalible")
        return buffers

def send_buffer_to_fit(name,buffer):
    url =f"http://{name}.local/ai/fit_data"
    response = requests.post(url,json=buffer)
    return response

def partition_and_send(candidates,partitioned_data):
    for cluster, data_amount in candidates.items():
        data_to_send = partitioned_data[:data_amount]
        # Remove the sent data from partitioned_data
        partitioned_data = partitioned_data[data_amount:]
        print(f"Sending partioned data to cluster:{cluster}",flush=True)
        send_buffer_to_fit(cluster, data_to_send)

def make_space(partioned_data,buffers):
    space_needed = len(partioned_data)

    # Contains the candidate and free space they have
    candidates_to_send_data = {}
    if buffers:
        for cluster in buffers:
             # Get how much data is in each cluster, if theres any free space, use it
             freespace_in_cluster = BUFFER_SIZE - len(buffers[cluster])
             if freespace_in_cluster > 0:
                 candidates_to_send_data[cluster] = freespace_in_cluster
                 space_needed = space_needed - freespace_in_cluster
                 if space_needed < 1:
                     break
        #If enough space in other clusters, send data on a candidate basis
        #Else also send data but give most to *lower cluster
        min_rank_cluster = max(buffers, key=lambda cluster: clusters[cluster])

        if not candidates_to_send_data:
            print(f"\: No free space avalible, sending all to min rank cluster {min_rank_cluster}",flush=True)
            candidates_to_send_data[min_rank_cluster] = 0
        if space_needed > 0:
            print(f"space_needed:{space_needed}")
            # Add un-allocate space to lowest ranked cluster
            candidates_to_send_data[min_rank_cluster]+=space_needed
        partition_and_send(candidates_to_send_data,partioned_data)
        print(f"Candidates to send extra data {candidates_to_send_data} \n Sending...",flush=True)
    else:
        print("No buffers avalible")

# ------------------------------------------------------------------------------------------------------------------------

def do_inference():
    global data_buffer
    unprocessed_input_data = None
    print("doing inference")

    while True:
        print("doing inference")
        unprocessed_input_data = None
        result=""
        original_cluster=""
        with data_lock:
            if data_buffer:
                unprocessed_input_data=data_buffer.pop(0)
                result = unprocessed_input_data["result"]
                del unprocessed_input_data["rank"]
                original_cluster = unprocessed_input_data["cluster_name"]
                del unprocessed_input_data["cluster_name"]
                del unprocessed_input_data["result"]
                raw = list(unprocessed_input_data.values())

        #Check if its been set
        if unprocessed_input_data:
            data_good=True
            for i in unprocessed_input_data.values():
                if i == None:
                    print("Missing data, infrence not possible")
                    data_good=False
                    break
            if data_good:
            # FEED IN THE INPUT DATA
                with training_lock:
                    node_data, prob_rain = inference_data(raw,loaded_model)
                if prob_rain > RAIN_THRESHOLD:
                    print("Edge model thinks its raining:",prob_rain,"mm",flush=True)
                else:
                    print("Edge model thinks its dry:",prob_rain,"mm",flush=True)
                send_metrics("edge_inference",[float(prob_rain[0])])
                send_metrics("edge_inference_vs_result",[abs(float(result)-float(prob_rain[0]))])
                send_to_control_model(node_data,result,original_cluster,raw)
        sleep(1)

def send_to_control_model(node_data,result,original_cluster,raw):
    node_list = node_data.tolist()
    sending_url = f"http://control.local/{original_cluster}/edge_result"
    if result == None:
        result = "empty"
    else:
        print("result:",result)
        send_metrics("actual_result",[result])
        result = str(float(result)*1000)
        
    # else:
    #     result = float(result # This shfits the decimal place up so fine data doesnt get lost when converting to decimal
    headers = {'Content-Type': 'application/json'}
    if SPLIT_CHECK:
        requests.post(sending_url,json={'node_data': node_list,'name':original_cluster,'result':result,'raw':raw},headers=headers)
    else:
        requests.post(sending_url,json={'node_data': node_list,'name':original_cluster,'result':result},headers=headers)
print("starting inference thread")
thread_inference = threading.Thread(target=do_inference)
thread_inference.daemon = True
thread_inference.start()

if __name__ == "__main__":
    while True:
        fetch_devices()
        sleep(INTERVAL)