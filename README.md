# 🌿 Plant Disease Classifier

A simple and interactive Streamlit web application for detecting plant diseases from leaf images using a trained TensorFlow model.

## 🚀 Features

- Upload a plant leaf image
- Automatically detects disease using deep learning
- Trained on various plant types including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## 🧠 Model

The model is a Convolutional Neural Network (CNN) trained using TensorFlow/Keras. It expects images resized to **128x128** pixels.

## 🖼️ Classes Covered

- Apple Scab
- Black Rot
- Cedar Apple Rust
- Apple Healthy
- Blueberry Healthy
- ... *(and many more)*

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run

```bash
streamlit run main.py
```

## ℹ️ Notes

- Disease diagnosis requires TensorFlow and a `trained_plant_disease_model.keras` file in the project folder.
