import paho.mqtt.client as mqtt
import random
import time

# MQTT Broker details
broker = "localhost"  # Replace with your EdgeX instance or cloud broker IP
port = 1883
topic = "devices/temperature"

# Simulated device ID
device_id = "sensor-001"

def simulate_device():
    client = mqtt.Client(device_id)
    client.connect(broker, port)
    print("Connected to MQTT broker")

    while True:
        # Generate random temperature
        temperature = round(random.uniform(20.0, 30.0), 2)
        payload = {
            "device_id": device_id,
            "temperature": temperature,
            "timestamp": time.time()
        }
    
        client.publish(topic, str(payload))
        print(f"Published: {payload}")
        time.sleep(5)  # Send data every 5 seconds

if __name__ == "__main__":
    simulate_device()
