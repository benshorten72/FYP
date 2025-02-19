from random import randint
from time import sleep
from typing import List
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify
import threading
import requests
import os
requests.packages.urllib3.util.connection.HAS_IPV6 = False

CLUSTER_NAME =os.getenv("CLUSTER_NAME")
CLUSTER_RANK =os.getenv("CLUSTER_RANK")

INTERVAL = 3 # Maybe env var
COLUMNS = ["sighting_date","location"] # env var 
BUFFER_SIZE = 10 # Env var 

PROFILE_URL = f"http://{CLUSTER_NAME}.local/core-metadata/api/v3/device/profile/name/Generic-MQTT-String-Float-Device"
CONTROL_URL = "http://control.local"
data_lock = threading.Lock()
# Incoming data added to buffer and is reset on interval
incoming_data = {}
listeners = []
data_buffer = []
client = mqtt.Client()
clusters = {}
sensors ={}
app = Flask(__name__)

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
    if (len(listeners)==len(COLUMNS)):
        if len(data_buffer) >= BUFFER_SIZE:
            data_buffer.pop(0)

        print(f"Current incoming data: {incoming_data}")
        with data_lock:
            processed_data = dict(incoming_data)
            processed_data["rank"] = CLUSTER_RANK
            data_buffer.append(dict(processed_data))
            #Reset incoming data after interval 
            for column in incoming_data.keys():
                incoming_data[column] = None
    
    if len(data_buffer) == BUFFER_SIZE:
        partioned_data = data_buffer[len(data_buffer)//2:]
        with data_lock:
            print("data buffer halved")
            data_buffer = data_buffer[:len(data_buffer)//2]
        make_space(partioned_data)


thread = threading.Thread(target=mqtt_thread)
thread.daemon = True 
thread.start()
# ---------------------------------------

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

if __name__ == "__main__":
    while True:
        fetch_devices()
        sleep(INTERVAL)
