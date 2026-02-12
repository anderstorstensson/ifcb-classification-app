import time
import torch
from PIL import Image
from main import model, image_transform

img = Image.open("data/models/ifcb-plankton-resnet50/examples/chaetoceros_didymus.png").convert("RGB")
t = image_transform(img).unsqueeze(0)

# warmup
with torch.no_grad():
    model(t)

start = time.perf_counter()
for _ in range(20):
    with torch.no_grad():
        model(t)
print(f"Avg: {(time.perf_counter() - start) / 20 * 1000:.1f} ms/image")
