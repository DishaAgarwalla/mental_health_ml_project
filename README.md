# 🧠 Mental Health Detection System

An AI-powered system to detect potential mental health concerns from text using machine learning.

## 📋 Features

- **Text Analysis**: Analyze statements for mental health indicators
- **Real-time Predictions**: FastAPI backend with instant results
- **Interactive Dashboard**: Streamlit frontend with visualizations
- **Prediction Logging**: SQLite database to track all analyses
- **Model Comparison**: Multiple algorithms with performance metrics
- **Confidence Scores**: Probability-based confidence indicators

## 🏗️ Project Structure
mental-health/
├── data/
│ └── Combined Data.csv
├── model/
│ ├── model.pkl
│ └── vectorizer.pkl
├── logs/
│ └── predictions.db
├── src/
│ ├── init.py
│ ├── preprocess.py
│ ├── train_model.py
│ ├── predict.py
│ ├── api.py
│ └── database.py
├── app.py
├── requirements.txt
└── README.md


## 🚀 Installation

1. **Clone the repository**
```bash
git clonehttps://github.com/DishaAgarwalla/mental_health_ml_project
 
