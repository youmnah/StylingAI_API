# AI Clothing Classification API (FastAPI)

## Overview

This project is a **multi-model AI system for clothing detection and classification** built using **FastAPI**.
It analyzes an uploaded clothing image and predicts multiple attributes such as:

* Category (Top / Bottom / Shoes / Accessories)
* Clothing type
* Pattern
* Color (RGB + name)
* Season suitability
* Style classification

The system combines multiple deep learning models and classical ML pipelines (TensorFlow, PyTorch, Hugging Face Transformers, Roboflow, and ONNX).

---

## Features

* 🧠 Multi-model ensemble inference pipeline
* 👕 Clothing category detection
* 🎨 Pattern classification
* 🌈 Color detection (RGB + named color)
* 🧥 Sub-category classification (Top / Bottom / Shoes / Accessories)
* 🌦️ Season prediction
* 👗 Style prediction
* 🔍 Roboflow object detection fallback
* ⚡ FastAPI REST endpoint for real-time inference

---

## Tech Stack

* Python 3.9+
* FastAPI
* TensorFlow / Keras
* PyTorch
* Hugging Face Transformers (ViT)
* OpenCV
* Roboflow API
* ONNX Runtime
* NumPy / PIL

---

## Project Structure

```
project/
│
├── main.py                  # FastAPI application
├── model/
│   ├── model_type/         # ViT model
│   ├── model_pattern/      # Pattern model (.json + .h5)
│   ├── model_top/
│   ├── model_bottom/
│   ├── model_shoes/
│   ├── model_sub/
│   └── model_rf/           # ONNX model
│
├── utils/
│   ├── color.py            # Color detection logic
│   ├── config.py           # Label mappings
│   ├── item_type.py        # Fine-tuning logic
│
└── README.md
```

---

## API Endpoint

### Predict Clothing Item

**POST** `/predict-item/`

### Request

* Form-data:

  * `file` → image file (jpg/png)

### Example (cURL)

```bash
curl -X POST "http://localhost:8000/predict-item/" \
-F "file=@image.jpg"
```

---

## Response Example

```json
{
  "category": "top",
  "type_Id": 2,
  "type": "shirt",
  "pattern_Id": 1,
  "pattern": "striped",
  "color": "blue",
  "color_rgb": [120, 180, 255],
  "season": "summer",
  "style": "casual"
}
```

---

## Model Pipeline

The prediction pipeline follows these steps:

1. **Pattern Model (CNN - Keras)**
2. **Color Detection (OpenCV + custom logic)**
3. **Type Classification (ViT - HuggingFace)**
4. **Roboflow Model (fallback detection)**
5. **Sub-category Model (TensorFlow)**
6. **Specialized Models (Top / Bottom / Shoes)**
7. **Fine-tuning logic (`fine_tuning`)**

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/clothing-ai-api.git
cd clothing-ai-api
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

```txt
fastapi
uvicorn
numpy
pandas
pillow
opencv-python
tensorflow
keras
torch
transformers
roboflow
onnxruntime
scikit-learn
matplotlib
seaborn
plotly
missingno
```

---

## Run the API

```bash
uvicorn main:app --reload
```

Then open:

```
http://127.0.0.1:8000/docs
```

---

## Notes

* Large models should NOT be pushed to GitHub (use `.gitignore`)
* Roboflow API key should be stored in `.env`
* GPU recommended for faster inference
* ONNX model used for optimized inference pipeline

---

## Security Warning

⚠️ Remove or hide this before publishing:

* Roboflow API key:

  ```python
  Roboflow(api_key="YOUR_KEY")
  ```

Use instead:

```python
import os
api_key = os.getenv("ROBOFLOW_API_KEY")
```

---

## Future Improvements

* Docker containerization
* Model versioning system
* Async batch prediction endpoint
* GPU inference optimization
* Frontend UI (React / Angular dashboard)

---

## Author

AI Clothing Classification System
Built with FastAPI + Deep Learning + Multi-model Ensemble
