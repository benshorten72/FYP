from flask import Flask, request, jsonify

app = Flask(__name__)

cluster_list = {}

@app.route('/add_cluster', methods=['POST'])
def add_cluster():
    data = request.get_json()
    name = data.get('name')
    rank = data.get('rank')
    if name in cluster_list:
        return jsonify({"error": "Name already exists"}), 409
    else:
        if name and rank:
            cluster_list[name] = rank
            return jsonify({"message": f"Name '{name}' added successfully!"}), 200
        else:
            return jsonify({"error": "No name and/or rank provided in the request"}), 400

@app.route('/get_clusters', methods=['GET'])
def get_clusters():
    return jsonify({"clusters": cluster_list}), 200

@app.route('/delete_cluster', methods=['POST'])
def delete_cluster():
    data = request.get_json()
    name = data.get('name')

    if name:
        if name in cluster_list:
            del cluster_list[name]
            return jsonify({"message": f"Name '{name}' deleted successfully!"}), 200
        else:
            return jsonify({"error": f"Name '{name}' not found in the list"}), 404
    else:
        return jsonify({"error": "No name provided in the request"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
