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
df = pd.read_csv(csv_file)
# If column name not defined, send data from all columns
if column_name != '':
    print("Column provided, reading all")
    topic = cluster_name+"/"+column_name
    column_data = df[column_name].tolist()

    client = mqtt.Client()

    client.connect(broker, port, 60)
    while True:
        for data in column_data:
            client.publish(topic, data)
            print("Publishing topic",topic,"of data",data,"to",broker)
            sleep(2)
        print("Reach end, looping")
else:
    print("No column provided, reading all")
    column_data = df.columns.tolist()
    client = mqtt.Client()
    client.connect(broker, port, 60)
    while True:
        for index, row in df.iterrows():
            for column in column_data:
                topic = f"{cluster_name}/{column}"  # Create the topic
                message = row[column]  # Get the data for the current column in the current row
                client.publish(topic, str(message))
                print(f"Publishing topic {topic} with data {message} to {broker}")
            sleep(2)
        print("Reached end, looping")