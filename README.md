# DynaDetect AI — Hidden Leaf Vision System

> **Advanced Zero-Shot Object Detection & Target Registry Portal**

DynaDetect AI is a dynamic computer vision application built to mimic the analytical precision of the Sharingan. By integrating state-of-the-art edge AI models with a high-performance backend, it provides real-time, zero-shot object detection capabilities through an immersive, themed web interface.

---

##  Key Features

* **Sharingan Vision Interface:** A highly responsive, custom-designed frontend dashboard built with HTML, CSS, and Vanilla JavaScript.
* **Zero-Shot Object Detection:** Utilizes **Grounding DINO** to detect and track any object purely based on text prompts—no rigid pre-training categories required.
* **Real-Time Analytics:** Integrates **YOLO** (You Only Look Once) for high-speed, standard object detection bounding boxes.
* **Target Registry System:** A dedicated UI section that logs and catalogs detected objects dynamically.
* **High-Speed API:** Backend powered by **FastAPI**, ensuring rapid inference times and smooth communication between the web portal and the AI models.

---

##  Tech Stack

**Frontend:**
* HTML5 / CSS3
* Vanilla JavaScript
* Custom Fonts (Orbitron, Rajdhani, Share Tech Mono)

**Backend & AI:**
* Python
* FastAPI
* YOLO (Ultralytics)
* Grounding DINO
* Uvicorn (ASGI Server)

---

##  Local Installation & Setup

Since this project relies on heavy AI models (YOLO, Grounding DINO) and a FastAPI backend, the web server needs to run locally to process the video/image feeds. 

### Prerequisites
* Python 3.8+
* Git
* A webcam (for real-time detection)

### 1. Clone the Repository
```bash
git clone [https://github.com/skjha3439/dynadetect-ai.git](https://github.com/skjha3439/dynadetect-ai.git)
cd dynadetect-ai
