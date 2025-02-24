from random import randint
from time import sleep
from typing import List
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify
import threading
import requests
import os
import numpy as np
from ai_edge_litert.interpreter import Interpreter

requests.packages.urllib3.util.connection.HAS_IPV6 = False

CLUSTER_NAME =os.getenv("CLUSTER_NAME")
CLUSTER_RANK =os.getenv("CLUSTER_RANK")

INTERVAL = 3 # Maybe env var
COLUMNS = ["ind","indt","temp","indw","wetb","dewpt","vappr","rhum","msl","indm","wdsp","indwd","wddir","ww","w","sun","vis","clht","clamt"] # env var 
BUFFER_SIZE = 10 # Env var 
RAIN_THRESHOLD = 0.5
MODEL_PATH = "/app/model/model.tflite"
PROFILE_URL = f"http://{CLUSTER_NAME}.local/core-metadata/api/v3/device/profile/name/Generic-MQTT-String-Float-Device"
CONTROL_URL = "http://control.local"
CONTROL_AI_URL = CONTROL_URL+f"/{CLUSTER_NAME}"
data_lock = threading.Lock()
# Incoming data added to buffer and is reset on interval
incoming_data = {}
listeners = []
data_buffer = []
client = mqtt.Client()
clusters = {}
sensors ={}
app = Flask(__name__)

@app.route('/edge_result', methods=['POST'])
def edge_result():
    json_data = request.json 
    if not json_data:
        return jsonify({"error": "No data provided"}), 400

@app.route('/fit_data', methods=['POST'])
def fit_data():
    global data_buffer
    try:
        data_to_be_fit = request.json 
        if not data_to_be_fit:
            return jsonify({"error": "No data provided"}), 400
        with data_lock:
            print("\n")
            print("*"*40)
            print("--- Recieving incoming data to fit ---")
            data_buffer.extend(data_to_be_fit)            
            sorted_data = sorted(data_buffer, key=lambda x: x["rank"])
            data_buffer = sorted_data[:10]
            print(f"Received buffer")
            print(f"Loosing data:",len(sorted_data[10:]))
            print("*"*40)
            print("\n")



   
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
        print(f"Sending partioned data to cluster:{cluster}")
        send_buffer_to_fit(cluster, data_to_send)


def make_space(partioned_data):
    space_needed = len(partioned_data)
    buffers = get_buffers()
    if not buffers:
        return
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
            print(f"\: No free space avalible, sending all to min rank cluster {min_rank_cluster}")
            candidates_to_send_data[min_rank_cluster] = 0
        if space_needed > 0:
            print(f"space_needed:{space_needed}")
            # Add un-allocate space to lowest ranked cluster
            candidates_to_send_data[min_rank_cluster]+=space_needed
        partition_and_send(candidates_to_send_data,partioned_data)
        print(f"Candidates to send extra data {candidates_to_send_data} \n Sending...")
    else:
        print("No buffers avalible")

# ------------------------------------------------------------------------------------------------------------------------
# AI
interpreter = Interpreter(model_path=MODEL_PATH)
signatures = interpreter.get_signature_list()
interpreter.allocate_tensors()

# GET THE OUTPUT DATA
interpreter.invoke()
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])

def inference():
    global data_buffer
    unprocessed_input_data = None
    print("doing inference")

    while True:
        print("doing inference")
        unprocessed_input_data = None
        with data_lock:
            if data_buffer:
                unprocessed_input_data=data_buffer.pop(0)
                del unprocessed_input_data["rank"]
                raw = list(unprocessed_input_data.values())
        #Check if its been set
        if unprocessed_input_data:
            # FEED IN THE INPUT DATA
            input_details = interpreter.get_input_details()
            input_data = np.array([raw], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            if output_data.shape[1] == 2: 
                prob_no_rain, prob_rain = output_data[0]
            else:  
                prob_rain = output_data[0][0]
                prob_no_rain = 1 - prob_rain

            if prob_rain > RAIN_THRESHOLD:
                print(f"It's raining! (Probability: {prob_rain:.2f})")
            else:
                print(f"It's not raining. (Probability: {prob_rain:.2f})")
            requests.post(CONTROL_AI_URL,prob_rain)
        sleep(5)
thread_inference = threading.Thread(target=inference)
thread_inference.daemon = True
thread_inference.start()

if __name__ == "__main__":
    while True:
        fetch_devices()
        sleep(INTERVAL)