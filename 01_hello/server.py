from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Docker backend with CORS enabled!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
