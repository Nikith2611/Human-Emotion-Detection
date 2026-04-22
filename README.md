# 🎭 Human Emotion Detection - Real-Time CNN Based System

## 📌 Overview

This project is a real-time Human Emotion Detection system that uses a Convolutional Neural Network (CNN) to classify facial expressions from live webcam input. It detects faces, predicts emotions, and visualizes results through an interactive dashboard.

The system demonstrates a complete deep learning pipeline:

* Data preprocessing
* Model training
* Model export and reuse
* Real-time inference
* Visualization and monitoring

---

## 🚀 Features

* 🎥 Real-time emotion detection using webcam
* 🧠 CNN-based deep learning model
* 🧍 Face detection using OpenCV Haar Cascade
* 📊 Live emotion count tracking
* 🥧 Dynamic pie chart visualization
* ⚡ Stabilized predictions using consecutive frame logic

---

## 🗂️ Project Structure

```id="p1v9kx"
├── .gitignore
├── Datasets.txt
├── LICENSE
├── README.md
├── Requirements.txt
├── trainmodel.ipynb
├── realtimedetection.py
```

---

## 🧠 Model Training (`trainmodel.ipynb`)

The Jupyter Notebook handles the full training pipeline.

### 🔹 Steps:

1. Load dataset from directory structure
2. Convert image paths and labels into a DataFrame
3. Preprocess images:

   * Convert to grayscale
   * Resize to 48×48
   * Normalize pixel values
4. Encode labels:

   * Label Encoding
   * One-hot Encoding
5. Build CNN model:

   * Multiple Conv2D + MaxPooling layers
   * Dropout for regularization
   * Dense layers for classification
6. Train the model using training dataset
7. Export trained model for reuse

### 💾 Model Export:

* `facialemotionmodel.h5` → Saved trained model
* Model can be reloaded and directly used for inference

---

## ⚡ Real-Time Detection (`realtimedetection.py`)

This script performs real-time emotion detection using the trained model.

### 🔹 Workflow:

1. Load trained model
2. Capture webcam frames using OpenCV
3. Detect faces using Haar Cascade
4. Extract face region (ROI)
5. Preprocess image (resize, normalize, reshape)
6. Predict emotion using CNN
7. Apply stabilization logic (reduces noisy predictions)
8. Update dashboard:

   * Emotion counts
   * Pie chart visualization

---

## 🧩 Tech Stack

* **Deep Learning**: TensorFlow, Keras
* **Computer Vision**: OpenCV
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib
* **UI Dashboard**: Tkinter

---

## ⚙️ Installation

```bash id="8i9z4k"
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
pip install -r Requirements.txt
```

---

## ▶️ Usage

### 🔹 Train the Model

```bash id="q8nl3w"
jupyter notebook trainmodel.ipynb
```

### 🔹 Run Real-Time Detection

```bash id="w9g6sx"
python realtimedetection.py
```

---

## 🧪 Output

* Real-time webcam feed with predicted emotions
* Emotion count dashboard
* Live emotion distribution pie chart

---

## 📈 Future Improvements

* Replace Haar Cascade with deep learning-based face detection (MTCNN, RetinaFace)
* Use transfer learning (MobileNet, ResNet)
* Add temporal modeling (LSTM) for better stability
* Optimize model for low-latency inference
* Deploy using AWS SageMaker or Azure ML

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit pull requests.

---

## 📜 License

This project is licensed under the terms of the LICENSE file.

---

## 👨‍💻 Author

Nikith Gokul
AI Engineer | Cloud & Generative AI Enthusiast

---
