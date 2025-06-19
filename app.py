# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from model import train_model, test_model
from predict import predict_image
import os

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    result = train_model()
    return jsonify({"message": result})

@app.route("/test", methods=["GET"])
def test():
    print("trening")
    result = test_model()
    return jsonify(result)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(file_path)
    result = predict_image(file_path)
    os.remove(file_path)
    return jsonify(result)

if __name__ == "__main__":
    print("Uruchamiam aplikacje Flask...")

    app.run(debug=True)
