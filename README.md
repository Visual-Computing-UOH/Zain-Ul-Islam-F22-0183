# Real-Time Object Detection using YOLOv4

## 📌 Introduction
This project performs real-time object detection using a pre-trained YOLOv4 model. It uses a webcam to detect and classify multiple objects in live video. The model is trained on the COCO dataset, so it can recognize common objects like persons, cars, bottles, chairs, etc.

---

## 🎯 Objectives
- Detect objects in real time using a webcam
- Use pre-trained YOLOv4 (no training required)
- Draw bounding boxes around detected objects
- Display object names along with confidence scores

---

## 🛠️ Technologies Used
- Python  
- OpenCV (cv2)  
- NumPy  

---

## 📂 Project Structure
project-folder/
│── yolov4.cfg  
│── yolov4.weights  
│── coco.names  
│── main.py  
│── README.md  

---

## ⚙️ Requirements
Make sure you have Python installed, then install required libraries:

pip install opencv-python numpy

---

## ▶️ How to Run
1. Download the following YOLOv4 files:
   - yolov4.cfg  
   - yolov4.weights  
   - coco.names  

2. Place all files inside the project folder

3. Run the program:

python main.py

4. Your webcam will open and start detecting objects

5. Press **q** on the keyboard to exit the program

---

## 📸 Output
- Live webcam video is displayed  
- Objects are detected in real time  
- Bounding boxes are drawn around objects  
- Each object shows:
  - Label (object name)  
  - Confidence score  

---

## ⚠️ Limitations
- Slow performance when running on CPU  
- Better hardware gives better speed  
- Detection accuracy depends on lighting conditions  
- Only detects objects from COCO dataset  

---

## 🚀 Future Improvements
- Use YOLOv4-tiny for faster detection  
- Add GPU (CUDA) support  
- Train model on custom dataset  
- Create a graphical user interface (GUI)  
- Add image and video file detection options  

---

## 👨‍💻 Author
Zain Ul Islam
BS Artificial Intelligence  
University Of Haripur

---

## 📌 Note
Make sure all YOLO files (cfg, weights, names) are in the same directory as the Python file before running the program.
