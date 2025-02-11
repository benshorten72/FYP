from time import sleep
import pandas as pd
import paho.mqtt.client as mqtt
import os

cluster_name=os.getenv("CLUSTER_NAME")
column_name =os.getenv("COLUMN_NAME")
file_name =os.getenv("FILE_NAME")
broker =os.getenv("MQTT_IP")

csv_file = "./data/"+file_name
port = 1883
topic = cluster_name+"/data"

df = pd.read_csv(csv_file)

column_data = df[column_name].tolist()


client = mqtt.Client()

client.connect(broker, port, 60)

for data in column_data:
    client.publish(topic, data)
    print("Publishing topic",topic,"of data",data,"to",broker)
    sleep(50)
for data in column_data:
    client.publish(topic, data)
    print("Publishing topic",topic,"of data",data,"to",broker)
    sleep(50)
for data in column_data:
    client.publish(topic, data)
    print("Publishing topic",topic,"of data",data,"to",broker)
    sleep(50)