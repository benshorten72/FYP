from flask import Flask, request
app = Flask(__name__)
@app.route('/', methods=['POST'])
def handle_post():
    data = request.json  # Get JSON data from the POST request
    print("Received data:", data)
    return {"message": "Data received"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)