import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Target correct directory name from your folder structure
model_dir_name = "distilled-student-final"
model_path = os.path.abspath(model_dir_name)

print(f"Loading distilled model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 2. Measure model directory size on disk
def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total / (1024 * 1024)

print(f"📦 Model Size on Disk: {get_dir_size(model_path):.2f} MB")

# 3. Benchmark Inference Speed & Throughput
sample_text = "This product works surprisingly well and exceeded my expectations!"
inputs = tokenizer(sample_text, return_tensors="pt")

# Warm-up run
with torch.no_grad():
    _ = model(**inputs)

# Measure average latency over 100 runs
num_runs = 100
start_time = time.perf_counter()

with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)

end_time = time.perf_counter()

total_time = end_time - start_time
avg_latency_ms = (total_time / num_runs) * 1000
throughput = num_runs / total_time

print(f"⏱️ Average Latency: {avg_latency_ms:.2f} ms")
print(f"🚀 Throughput: {throughput:.2f} requests/sec")