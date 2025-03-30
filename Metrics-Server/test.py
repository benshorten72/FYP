from time import sleep
from flask import Flask, request, jsonify, render_template
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

data_lock = threading.Lock()
app = Flask(__name__)
PORT = os.getenv("PORT")
METRICS_LIST=[]
# DICT -> Data_name -> cluster -> time
metrics_dict={}
@app.route('/ui')
def index():
    return render_template('index.html')

@app.route('/add_metrics', methods=['POST'])
def add_metrics():
        global metrics_dict
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        if "cluster_name" not in data:
            return jsonify({"error": "Missing 'cluster_name'"}), 400
        if "data_name" not in data:
            return jsonify({"error": "Missing 'data_name'"}), 400
        if "values" not in data:
            return jsonify({"error": "Missing 'values'"}), 400
        cluster_name = data["cluster_name"]
        data_name = data["data_name"]
        time = data["time"]
        values= data["values"]

        if len(values)==1:
            add_time_series_data(metrics_dict, cluster_name,data_name,time,values)
        logging.info(f"Metrics received from {data['cluster_name']}: {data['data_name']}")
        return jsonify({"status": "success", "message": "Metrics added successfully"}), 200

@app.route('/get_data_names', methods=['GET'])
def get_data_names():
    global metrics_dict
    data_names = list(metrics_dict.keys())
    return data_names

@app.route('/get_metrics/<data_name>', methods=['GET'])
def get_metrics(data_name):
    global metrics_dict
    try:
        if data_name not in metrics_dict:
            return jsonify({"error": f"Data name '{data_name}' not found"}), 404
        metrics = metrics_dict[data_name]
        return jsonify(metrics), 200

    except Exception as e:
        logging.error(f"Error fetching metrics: {e}")
        return jsonify({"error": "Internal server error"}), 500
@app.route('/get_cluster_data/<cluster_name>/<data_name>', methods=['GET'])
def get_cluster_data(cluster_name, data_name):
    global metrics_dict
    try:
        if data_name not in metrics_dict or cluster_name not in metrics_dict[data_name]:
            return jsonify({"error": f"Data for cluster '{cluster_name}' and data name '{data_name}' not found"}), 404
        return jsonify(metrics_dict[data_name][cluster_name]), 200
    except Exception as e:
        logging.error(f"Error fetching cluster data: {e}")
        return jsonify({"error": "Internal server error"}), 500
def add_time_series_data(metrics_dict,cluster,data_name,time,values):
    if data_name not in metrics_dict.keys():
        metrics_dict[data_name]={}
    if cluster not in metrics_dict[data_name].keys():
        metrics_dict[data_name][cluster]=[]
    metrics_dict[data_name][cluster].append([values[0],time])
    print("time series data added",metrics_dict,flush=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)