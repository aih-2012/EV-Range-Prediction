# EV Range Prediction Model 🚗🔋

A machine learning project that predicts the driving range of electric vehicles (EVs) based on various input features such as battery capacity, motor power, vehicle weight, and efficiency parameters.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview
This project aims to provide a predictive model for estimating the range of electric vehicles.  
It can help users, manufacturers, and enthusiasts **estimate the driving distance** of an EV under different conditions.

---

## Dataset
- Title: `Electric Vehicle Specs Dataset (2025)`
- Author: `Urvish Ahir`
- Link: `https://www.kaggle.com/datasets/urvishahir/electric-vehicle-specifications-dataset-2025`
- Number of entries: `1000`
- Columns include `[Battery Capacity, Motor Power, Drivetrain, Efficiency, etc.]`

---

## Features
The model uses the following features to predict EV range:
- Top Speed (Kmh)
- Battery Capacity (kWh)
- Number of cells
- Torque (nM)
- Acceleration (0-100, s)
- Fast charging Power (kW/dc)
- Towing capacity (kg)
- Motor Power (kW)
- Cargo Volume (L)
- Drivetrain (FWD/RWD/AWD)
- Segment
- Body type
- Vehicle Weight (kg)
- Efficiency (Wh/km)

---

## Model
- Model type: `Random Forest`
- Programming language: Python
- Libraries: `pandas`, `numpy`, `scikit-learn`, `[Optional: matplotlib, seaborn]`

## Usage
1. Clone the repository:
```
git clone https://github.com/yourusername/ev-range-predictor.git
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Run the prediction script:
```
python predict_range.py
```

4. Input vehicle parameters and get the predicted range.

## Evaluation Metrics
The model is evaluated using standard regression metrics:

- Mean Absolute Error (MAE): [14.125]
- Root Mean Squared Error (RMSE): [Insert value]
- R² Score: [Insert value]

### Example Code Snippet
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
# Future Work
- Include more features such as driving conditions, temperature, and load.
- Improve model accuracy using advanced algorithms or neural networks.
- Deploy as a web app or mobile app for interactive predictions.

# License
This project is licensed under the Creative Commons Attribution 4.0 International License.
