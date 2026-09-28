import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

model_path = os.path.abspath("distilled-student-final")
onnx_folder = os.path.abspath("onnx_distilled_model")
quantized_folder = os.path.abspath("onnx_quantized_model")

# 1. Export model directly to ONNX format via Optimum
print("🔄 Exporting PyTorch model to ONNX...")
model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained(onnx_folder)
tokenizer.save_pretrained(onnx_folder)
print(f"✅ Exported ONNX model to '{onnx_folder}'\n")

# 2. Apply INT8 Dynamic Quantization
print("⚡ Quantizing ONNX model to INT8...")
quantizer = ORTQuantizer.from_pretrained(onnx_folder)
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

quantizer.quantize(
    save_dir=quantized_folder,
    quantization_config=qconfig,
)
print(f"✅ Quantized model saved to '{quantized_folder}'\n")

# 3. Size Comparison
fp32_size = os.path.getsize(os.path.join(onnx_folder, "model.onnx")) / (1024 * 1024)
int8_size = os.path.getsize(os.path.join(quantized_folder, "model_quantized.onnx")) / (1024 * 1024)

print(f"📦 FP32 ONNX Size:     {fp32_size:.2f} MB")
print(f"📦 Quantized INT8 Size: {int8_size:.2f} MB")
print(f"📉 Size Reduction:     {((fp32_size - int8_size) / fp32_size) * 100:.1f}%\n")

# 4. Benchmark Quantized ONNX Runtime Speed
quantized_onnx_file = os.path.join(quantized_folder, "model_quantized.onnx")
session = ort.InferenceSession(quantized_onnx_file)

# Get valid input names expected by the ONNX model
expected_inputs = {input_node.name for input_node in session.get_inputs()}

sample_text = "This product works surprisingly well and exceeded my expectations!"
inputs = tokenizer(sample_text, return_tensors="np")

# Filter inputs to only include keys expected by the ONNX graph
onnx_inputs = {
    k: v.astype(np.int64) for k, v in inputs.items() if k in expected_inputs
}

# Warm-up run
_ = session.run(None, onnx_inputs)

# Measure average latency over 100 runs
num_runs = 100
start_time = time.perf_counter()
for _ in range(num_runs):
    _ = session.run(None, onnx_inputs)
end_time = time.perf_counter()

avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
throughput = num_runs / (end_time - start_time)

print(f"⏱️ Quantized ONNX Latency: {avg_latency_ms:.2f} ms")
print(f"🚀 Quantized Throughput:   {throughput:.2f} requests/sec")