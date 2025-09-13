# 🧠 ML Mini Project – Backend Cold Start Monitor with KNN Clustering and Analysis

[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://gunicorn.org/)
[![Render](https://img.shields.io/badge/Render-000000?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

---

<div align="center">

### 🚀 Live Demo

<a href="https://ml-mini-project-ehri.onrender.com" target="_blank">
  <img src="https://img.shields.io/badge/%20Deployed-FF4088?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo"/>
</a>
<img width="1912" height="946" alt="image" src="https://github.com/user-attachments/assets/68578406-2958-492a-be76-355cfb6b9206" />
<img width="1911" height="728" alt="image" src="https://github.com/user-attachments/assets/a122ac81-e5c0-438f-827b-bace5f62e8b0" />


</div>

---

## 📌 Project Overview

This project is a **Flask-based machine learning web app** that:
- Loads a pre-trained **KMeans clustering model** and **scaler**
- Accepts input data and performs clustering
- Provides REST API endpoints for **health check** and **data analysis**
- Serves a simple **frontend UI** via Flask templates

It is deployed on **Render** with Gunicorn for production.

---

## ⚙️ Tech Stack

- **Backend**: Flask + Gunicorn  
- **Machine Learning**: Scikit-Learn, Pandas, NumPy  
- **Frontend**: HTML, CSS, JavaScript (served via Flask)  
- **Deployment**: Render  

---

## 📂 Folder Structure
```
├── app.py # Main Flask app
├── requirements.txt # Dependencies
├── Procfile # For Gunicorn on Render
├── kmeans_model.pkl # Trained ML model
├── scaler.pkl # Fitted scaler
├── templates/
│ └── index.html # Frontend
├── static/ # CSS / JS files
└── analysis_data.csv # Sample dataset
```
---

## ▶️ Running Locally

1. **Clone this repo**:

```bash
git clone https://github.com/your-username/ML-MINI-PROJECT.git
cd ML-MINI-PROJECT
```
Create a virtual environment & activate:
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the Flask app:
```bash
python app.py
```
Visit the app: http://127.0.0.1:5000

### ☁️ Deployment on Render
This app is deployed on Render.

### Key Settings:

# Build Command: 
```
pip install -r requirements.txt
```
# Start Command: 
```bash
gunicorn app:app
```
# Environment: Python 3.x


### 📡 API Endpoints
GET / → Serves frontend UI

POST /ping_url → Health check endpoint

POST /analyze_data → Runs clustering on dataset

### ✨ Future Improvements
- Interactive frontend for uploading datasets

- More ML models (classification, regression)

- User authentication for saving results

