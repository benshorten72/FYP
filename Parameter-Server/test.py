from time import sleep
from flask import Flask, request, jsonify
import threading
import requests
import os
import numpy as np
import sys
requests.packages.urllib3.util.connection.HAS_IPV6 = False
import logging
import gzip
import base64
# Disable Flask/Werkzeug logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CLUSTER_NAME = os.getenv("CLUSTER_NAME")
CLUSTER_RANK = os.getenv("CLUSTER_RANK")
PORT = os.getenv("PORT")
INTERVAL = 3  # Maybe env var
CONTROL_URL = "http://control.local"
MAX_DATASET_SIZE = 1000  
BATCH_SIZE = 16

data_lock = threading.Lock()
app = Flask(__name__)
app.logger.disabled = True
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
#This is to make sure that if parameter server has no federated values it wont try and
#update a cluster (IE first one created) until it has calculated federated average
has_been_set = False
print(f"Max content length: {app.config['MAX_CONTENT_LENGTH']} bytes")

federated_averages = None
print(requests.get("http://control.local/control/get_clusters"))
print("Port:",PORT)

@app.route('/update_my_weights', methods=['POST'])
def update_my_weights():
    global federated_averages,has_been_set
    if has_been_set:
        json_data = request.get_json(force=True)
        cluster_name = json_data["name"]
        set_cluster_weights([cluster_name],federated_averages)
        return jsonify({"message": "Weights sent"}), 200 
    else:
        print("Cannot update clusters weights if federated averages are not calculated yet")
        return jsonify({"error": "Weights not calculated yet"}), 503 

@app.route('/get_weights', methods=['GET'])
def get_weights():
    #Provide a get for clusters when they get created
    try:
        print("Answering weight request from clusters", flush=True)
        if federated_averages:
            weights_json = [w.tolist() for w in federated_averages]
            return jsonify({"weights": weights_json}), 200
        else:
            return jsonify({"error": "Weights not calculated yet"}), 503 

    except Exception as e:
        return jsonify({"error": str(e)}), 500
def get_clusters():
    clusters = []
    response = requests.get(CONTROL_URL + "/control/get_clusters",timeout=1)
    data = response.json()
    for name, rank in data["clusters"].items():
            clusters.append(name)
    print("clusters retrieved",clusters)
    return clusters

def get_weights_from_clusters(clusters):
    weights = {}
    number_of_datas = {}
    valid_clusters = [] 

    for cluster in clusters:
        try:
            json_data = requests.get(f"http://control.local/{cluster}/get_weights", timeout=1)
            if json_data.status_code == 200:
                json_data = json_data.json()
                if json_data and "weights" in json_data and json_data["weights"]!=None:
                    weights[cluster] = [np.array(w) for w in json_data["weights"]]
                    number_of_datas[cluster] = json_data.get("data_amount", 0)
                    valid_clusters.append(cluster)  # Add only valid clusters
                else:
                    print(f"Weights missing from {cluster}")
            else:
                
                print(f"Weights not retrieved from cluster {cluster}",json_data.status_code,json_data.json()["error"])
        except Exception as e:
            print(f"Error retrieving weights from {cluster}: {e}")

    return weights, number_of_datas, valid_clusters  


def federated_average(weights,number_of_datas,clusters):
    global has_been_set
    print("Getting federated average")
    print(weights[clusters[0]])
    federated_average_weight = np.zeros_like(weights[clusters[0]])
    total = sum(number_of_datas.values()) 
    if total == 0:
        total=1
    print(clusters)
    for cluster in clusters:
        current_weights =weights[cluster]
        current_number = number_of_datas[cluster]
        if current_number == 0:
            current_number=1
        federated_average_weight += np.array(current_weights) * (current_number / total)
    has_been_set=True
    return federated_average_weight

def set_cluster_weights(clusters, federated_averages):
    for cluster in clusters:
        size_mb = sum(arr.nbytes for arr in federated_averages) / (1024 * 1024)
        print(f"Size of federated_averages in numpy: {size_mb:.2f} MB")
        weights_json = [w.tolist() for w in federated_averages]
        size_mb = sys.getsizeof(weights_json) / (1024 * 1024)
        response = requests.post(f"http://control.local/{cluster}/set_weights", json={"weights": weights_json})
        print(response.status_code)
        print(f"{cluster} weights have been federated")
        weights_json = [w.tolist() for w in federated_averages] 

# Start the Flask app in a separate thread, so if app while sleeping gets pinged to
# set a new clusters weights it can

def run_flask_app():
    app.run(host='0.0.0.0', port=PORT)

flask_thread = threading.Thread(target=run_flask_app)
flask_thread.daemon = True
flask_thread.start()
# Keep the main thread alive to keep the daemon threads running
while True:
    clusters = get_clusters()
    if clusters:
        weights, number_of_datas, clusters = get_weights_from_clusters(clusters)
        print("Current number of weights",len(weights))
        if weights:
            federated_averages = federated_average(weights, number_of_datas,clusters)
            set_cluster_weights(clusters,federated_averages)
        else:
            print("No weights present")
    else:
        print("No clusters present or control")
    sleep(10)