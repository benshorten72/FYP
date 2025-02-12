from random import randint
from time import sleep
import paho.mqtt.client as mqtt
from flask import Flask, request
import threading
import requests
import os

CLUSTER_NAME =os.getenv("CLUSTER_NAME")
INTERVAL = 8 # Maybe env var
COLUMNS = ["sighting_date"] # env var 
BUFFER_SIZE = 10 # Env var 

URL = f"http://{CLUSTER_NAME}.local/core-metadata/api/v3/device/profile/name/Generic-MQTT-String-Float-Device"

data_lock = threading.Lock()
# Incoming data added to buffer and is reset on interval
incoming_data = {}
listeners = []
data_buffer = []
client = mqtt.Client()


for i in COLUMNS:
    incoming_data[i]=None
    
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



sensors ={}

# I need to now
# 1. Get all devices periodcally
# 2. Create new mqtt subscriber for devices if it doesnt exist
# 3. Create a buffer to recieve and compile data within interval into example

def fetch_devices():
    try:
        response = requests.get(URL)
        response.raise_for_status() 
        devices = response.json()
        for device in devices["devices"]:
            device_name = device["name"]
        profile_name = device["profileName"]
        topic = device["protocols"]["mqtt"]["topic"]  # Extract the MQTT topic

        print(f"Device Name: {device_name}")
        print(f"Profile Name: {profile_name}")
        print(f"MQTT Topic: {topic}")

        # Check if data name (profile_name) is in column list
        if device_name in COLUMNS and device_name not in sensors.keys():
            print("This works")
            listeners.append(Listener(topic,device_name))     
            sensors[device_name] = listeners
        print("-" * 40)
        handle_incoming_data()
        print(f"Buffer has been updated to contain: {data_buffer}")
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
if __name__ == "__main__":
    while True:
        fetch_devices()
        sleep(INTERVAL)
