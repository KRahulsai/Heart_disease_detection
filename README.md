# Heart Disease Detection using Classification Algorithms

A machine-learning powered web application that predicts whether a patient has heart disease. The system dynamically adapts to **any** heart-disease CSV dataset — no hard-coded column names.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-green?logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)

---

## 📁 Project Structure

```
heart-disease-detection/
│
├── data/
│   └── heart.csv              # Your dataset (user provided)
│
├── backend/
│   ├── __init__.py
│   ├── app.py                 # Flask API server
│   ├── predict.py             # Prediction logic
│   └── model_loader.py        # Loads saved model artifacts
│
├── frontend/
│   └── app.py                 # Streamlit web UI
│
├── model/                     # Generated after training
│   ├── model.pkl              # Best trained model
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── encoder.pkl            # LabelEncoders for categorical features
│   ├── target_encoder.pkl     # Target encoder (if target is categorical)
│   ├── metadata.json          # Feature names, types, results
│   ├── model_comparison.png   # Performance comparison chart
│   ├── correlation_heatmap.png
│   └── target_distribution.png
│
├── notebook/
│   └── training.ipynb         # Jupyter notebook (optional)
│
├── train.py                   # Training pipeline script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/heart-disease-detection.git
cd heart-disease-detection
pip install -r requirements.txt
```

### 2. Add Your Dataset

Place your CSV file in the `data/` folder:

```
data/heart.csv
```

> The system auto-detects the target column. Columns named `target`, `label`, `class`, `output`, `diagnosis`, etc. are detected automatically. Otherwise, the last column is used by default.

### 3. Train the Model

```bash
python train.py --data data/heart.csv
```

**Options:**
| Flag | Description |
|------|-------------|
| `--data` | Path to your CSV dataset (required) |
| `--target` | Manually specify the target column name |

This will:
- Run exploratory data analysis
- Handle missing values & encode features
- Train 4 models (Logistic Regression, Random Forest, SVM, KNN)
- Evaluate and auto-select the best model
- Save all artifacts to `model/`

### 4. Start the Backend

```bash
python -m backend.app
```

The Flask API starts at `http://localhost:5000`.

**API Endpoints:**

| Method | Endpoint     | Description                        |
|--------|--------------|------------------------------------|
| GET    | `/health`    | Health check                       |
| GET    | `/metadata`  | Model metadata (features, classes) |
| POST   | `/predict`   | Send features, get prediction      |

**Example `POST /predict` request:**
```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

**Response:**
```json
{
  "status": "ok",
  "prediction": 0,
  "label": "0",
  "probability": [0.87, 0.13]
}
```

### 5. Start the Frontend

Open a **new terminal** and run:

```bash
streamlit run frontend/app.py
```

The Streamlit UI opens at `http://localhost:8501`.

---

## 🧠 Models Used

| Algorithm             | Description                                      |
|-----------------------|--------------------------------------------------|
| Logistic Regression   | Linear classifier for binary outcomes            |
| Random Forest         | Ensemble of decision trees                       |
| Support Vector Machine| Finds optimal hyperplane for classification      |
| K-Nearest Neighbors   | Instance-based learning using k closest samples  |

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

The model with the **highest F1-Score** is automatically selected and saved.

---

## 🔧 Using a Custom Dataset

This project is **dataset-agnostic**. To use any heart disease CSV:

1. Place your CSV in `data/`
2. Run training:
   ```bash
   python train.py --data data/your_dataset.csv --target your_target_column
   ```
3. Restart the backend and frontend
4. The UI will auto-generate input fields matching your dataset columns

---

## ☁️ Deployment (Render)

### Backend (Flask API)

1. Create a new **Web Service** on [Render](https://render.com)
2. Connect your GitHub repo
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python -m backend.app`
   - **Environment Variables:**
     - `PORT` = `5000`

### Frontend (Streamlit)

1. Create another **Web Service** on Render
2. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
   - **Environment Variables:**
     - `API_URL` = `https://your-backend.onrender.com`

---

## 📋 Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
