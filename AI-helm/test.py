from random import randint
from time import sleep
from typing import List
import paho.mqtt.client as mqtt
from flask import Flask, request
import threading
import requests
import os
requests.packages.urllib3.util.connection.HAS_IPV6 = False

CLUSTER_NAME =os.getenv("CLUSTER_NAME")
CLUSTER_RANK =os.getenv("CLUSTER_RANK")

INTERVAL = 8 # Maybe env var
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

for i in COLUMNS:
    incoming_data[i]=None

def get_clusters():
    
    try:
        response = requests.get(CONTROL_URL+"/control/get_clusters")
        response.raise_for_status() 
        data = response.json()
        print("Response JSON:", data)
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return
    
    if "clusters" in data:
        clusters={}
        # Add each cluster to the dictionary
        for cluster in data["clusters"]:
            name = cluster["name"]
            rank = cluster["rank"]
            if name != CLUSTER_NAME:
                clusters[name] = rank

def mqtt_thread():
    def on_message(client, userdata, message):
        print(f"Received data on topic: {message.topic}")
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
        print(f"Received from {self.name}: {data}")

    def unsubscribe(self):
        client.unsubscribe(self.topic)



sensors ={}

# I need to now
# 1. Get all devices periodcally
# 2. Create new mqtt subscriber for devices if it doesnt exist
# 3. Create a buffer to recieve and compile data within interval into example
def fetch_devices():
    try:
        #Fetch clusters
        get_clusters()

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

    except requests.exceptions.RequestException as e:
        print(f"Error fetching devices: {e}")

def handle_incoming_data():
    # Check if number of connected devices is equal to COLUMNS and that buffer size isnt overflowing
    print(f"Current listeners: {len(listeners)}")
    if (len(listeners)==len(COLUMNS)):
        if len(data_buffer) >= BUFFER_SIZE:
            data_buffer.pop(0)

        print(f"Current incoming data: {incoming_data}")
        with data_lock:
            data_buffer.append(dict(incoming_data))
            #Reset incoming data after interval 
            for column in incoming_data.keys():
                incoming_data[column] = None


thread = threading.Thread(target=mqtt_thread)
thread.daemon = True 
thread.start()
# ---------------------------------------

def get_buffers():
    buffers={}
    if clusters:
        for cluster in clusters:
            if clusters[cluster] < CLUSTER_RANK:
                response = requests.get(CONTROL_URL+"/get_buffer")
                response.raise_for_status() 
                data = response.json()
                
    else:
        print("No other clusters avalible")
        return buffers

def make_space():
    buffers = get_buffers()
    if buffers:
        return 
    else:
        print("No buffers avalible")
if __name__ == "__main__":
    while True:
        fetch_devices()
        sleep(INTERVAL)
