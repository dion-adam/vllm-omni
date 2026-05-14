# client.py
import base64
import io
import json
import time

from PIL import Image
import requests


def dummy_image_b64():
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


payload = {
    "task": "pick up the red cube",
    "state": [0, 0, 0, 0, 0, 0, 0],
    "images": {
        "image0": dummy_image_b64(),
        "image1": dummy_image_b64(),
        "image2": dummy_image_b64(),
    },
}

start = time.perf_counter()

r = requests.post(
    "http://localhost:8091/v1/actions/generations",
    json=payload,
)

end = time.perf_counter()

latency_ms = (end - start) * 1000

print(f"Latency: {latency_ms:.2f} ms")
print(f"Status: {r.status_code}")


