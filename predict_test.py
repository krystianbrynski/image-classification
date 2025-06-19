import requests

url = "http://127.0.0.1:5000/predict"
file_path = r"C:\Users\kryst\Source\Repos\image-classification\dataset\test\forest\20056.jpg"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
