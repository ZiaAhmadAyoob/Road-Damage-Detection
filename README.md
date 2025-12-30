# 🚧 Road Damage Detection using YOLOv11 & Streamlit

A deep learning–based computer vision application that detects road damages such as **potholes, cracks, and patches** from images and videos. The system uses **YOLOv11** for object detection and is deployed as a **Streamlit web application** for easy interaction.

---

## 🔍 Problem Statement
Poor road conditions like potholes and cracks pose serious risks to vehicles and pedestrians. Manual inspection is time-consuming and inefficient. This project aims to automate road damage detection using computer vision to support smart city and infrastructure monitoring systems.

---

## 🎯 Project Objectives
- Detect multiple types of road damage in real-world images
- Provide real-time inference through a web interface
- Build an end-to-end pipeline from dataset preparation to deployment

---

## 🧠 Model & Dataset
- **Model:** YOLOv11 (Ultralytics)
- **Task:** Object Detection
- **Classes:** Pothole, Road Crack, Patch
- **Dataset Format:** COCO → converted to YOLO format
- **Training:** Custom-trained on road damage dataset

---

## ⚙️ Tech Stack
- Python
- YOLOv11 (Ultralytics)
- OpenCV
- Streamlit
- NumPy, Pillow

---

## 🖥️ Web Application (Streamlit)
The deployed web app allows users to:
- Upload road images
- Detect and visualize road damages with bounding boxes
- Run inference using a trained YOLOv11 model

👉 **Live Demo:** _Add your Streamlit Cloud URL here_

---

## 📊 Model Performance
- **Accuracy (mAP@0.5):** ~89%
- **Precision:** ~87%
- **Recall:** ~83%

> Metrics may vary depending on dataset split and confidence thresholds.

---

## 📁 Project Structure
RoadDamage_App/
├── app.py
├── best.pt
├── requirements.txt
├── README.md


---
