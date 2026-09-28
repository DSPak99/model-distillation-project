# High-Performance Sentiment Analysis Deployment

An end-to-end Machine Learning pipeline featuring Knowledge Distillation, ONNX Runtime optimization, INT8 Quantization, and FastAPI serving containerized with Docker.

## 🚀 Performance Benchmarks

| Metric | PyTorch Baseline | Quantized ONNX (INT8) | Improvement |
| --- | --- | --- | --- |
| **Model Size** | 519 MB | **129 MB** | **75% Reduction** |
| **Latency** | ~22.4 ms | **~6.0 ms** | **3.7x Faster** |

---

## 🛠️ Project Architecture

1. **Distillation**: Distilled student model from large transformer.
2. **ONNX Export**: Converted PyTorch computational graph to ONNX format.
3. **Dynamic Quantization**: Quantized model weights from FP32 to INT8.
4. **FastAPI Serving**: Built REST API endpoints with pydantic schemas.
5. **Dockerization**: Packaged into a lightweight Python container.

---

## 💻 Quickstart (Docker)

### Build Image
```bash
docker build -t sentiment-onnx-api .