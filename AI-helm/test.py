from random import randint
from time import sleep
import paho.mqtt.client as mqtt
from flask import Flask, request
import threading
import requests

while(True):
    requests.post("http://why.local/ai/add-partner/pleaseee", json={"from":"me"})
    sleep(5)

@app.route('/', methods=['POST'])
def handle_post():
    data = request.json
    print("Received data:", data)
    return {"message": "Data received"}, 200

@app.route('/add-partner/<partner_name>', methods=['POST'])
def add_partner(partner_name):
    if partner_name not in partners:
        partners.append(partner_name)
        print("Partner added:", partner_name)
        return {"message": "Partner added"}, 200
    else:
        return {"message": "Partner already exists"}, 400

@app.route('/delete-partner/<partner_name>', methods=['POST'])
def delete_partner(partner_name):
    if partner_name in partners:
        partners.remove(partner_name)
        print("Partner deleted:", partner_name)
        return {"message": "Partner deleted"}, 200
    else:
        return {"message": "Partner doesn't exist"}, 400

@app.route('/mqtt-data', methods=['POST'])
def get_mqtt_data():
    data = request.json
    print("MQTT Data:", data)
    return {"message": "MQTT data received"}, 200

def on_mqtt_message(client, userdata, msg):
    for partner in partners:
        address = f"http://{partner}.local/ai"
        send_mqtt_data(address, msg.payload.decode())

def send_mqtt_data(address, message):
    try:
        response = requests.post(f"{address}/mqtt-data", json={"from": name, "message": message})
        print(f"Sent data to {address}: {response.status_code}")
    except requests.RequestException as e:
        print(f"Failed to send data to {address}: {e}")

def run_mqtt_subscriber():
    client = mqtt.Client()
    client.on_message = on_mqtt_message
    client.connect("edgex-mqtt-broker", 1883, 60)
    client.subscribe("result") 
    client.loop_forever()

mqtt_thread = threading.Thread(target=run_mqtt_subscriber)
mqtt_thread.daemon = True
mqtt_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
