import requests
import time

url = "http://localhost:8000/api/detect"
image_path = "/home/faraz/cooperative-foundational-models/custom_image.jpg"

print(f"Testing API with {image_path}...")
start_time = time.time()

try:
    with open(image_path, "rb") as f:
        response = requests.post(url, files={"file": f})
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Keys returned:", data.keys())
        if "metrics" in data:
            print("Metrics:", data["metrics"])
        if "error" in data:
            print("Server returned internal error:", data["error"])
        if "detections" in data:
            print(f"Num detections: {len(data['detections'])}")
            for det in data["detections"][:5]:
                print(f"  - {det['label']} ({det['confidence']:.2f}) - {det['type']}")
    else:
        print("Failed:", response.text)
except Exception as e:
    print("Error connecting to API:", str(e))

print(f"Total test time: {time.time() - start_time:.2f}s")
