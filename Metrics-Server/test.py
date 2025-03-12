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

data_lock = threading.Lock()
app = Flask(__name__)
PORT = os.getenv("PORT")
METRICS_LIST=[]
metrics_dict={}

def add_metrics():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        if "cluster_name" not in data:
            return jsonify({"error": "Missing 'cluster_name'"}), 400
        if "data_name" not in data:
            return jsonify({"error": "Missing 'data_name'"}), 400
        if "values" not in data:
            return jsonify({"error": "Missing 'values'"}), 400
        print(data,flush=True)
        logging.info(f"Metrics received from {data['cluster_name']}: {data['data_name']}")
        return jsonify({"status": "success", "message": "Metrics added successfully"}), 200

    except Exception as e:
        logging.error(f"Error processing metrics: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)