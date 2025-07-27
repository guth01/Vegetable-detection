# 🎯 YOLOv8 Object Detection Web App

A lightweight **single-page application** built using **Node.js**, **Express**, and plain **HTML/CSS/JS**, featuring real-time object detection powered by a **custom-trained YOLOv8 model**.

---

## 🚀 Features

- 🧠 Uses a **YOLOv8 model trained in Roboflow workspace**
- 🔍 Performs object detection using **OpenCV + Ultralytics YOLOv8**
- 📷 Accepts image input via browser and returns predictions with bounding boxes
- 🖥️ Simple frontend built with HTML, CSS, and JavaScript
- ⚙️ Node.js backend invokes a Python script to run the detection

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript (Single Page)
- **Backend**: Node.js + Express
- **Detection Pipeline**: Python (YOLOv8 + OpenCV)
- **Model Training**: Roboflow workspace (exported YOLOv8 weights)

---

## 🧠 How It Works

1. The dataset is uploaded and trained using **Roboflow**, then exported as YOLOv8 format.
2. The trained weights are loaded in a **Python script** that uses:
   - **Ultralytics YOLOv8** library
   - **OpenCV** for image handling
3. The Node.js server accepts image uploads and invokes the Python script.
4. The Python script performs inference and returns predictions.
5. The frontend displays results — annotated image or detection summary.

---

## 📦 Status

- Model training complete ✅  
- Image upload and prediction pipeline working ✅  
- UI is minimal but functional ✅  
- Not yet deployed 🚧

---

## 📌 Notes

- Roboflow was used only for dataset creation and training.
- All inference happens **locally** using the exported YOLOv8 model.
- The Python script is invoked from the Express server using `child_process`.

