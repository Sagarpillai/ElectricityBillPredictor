# ⚡ Electricity Bill Predictor - AI & ML Hackathon

## 📌 Problem Statement
This project addresses the challenge of predicting monthly electricity bills for households or facilities using appliance usage, city, company, and tariff rate data.

Built for the IIITDM AI & ML Hackathon, this solution helps energy providers forecast costs and detect anomalies efficiently.

---

## 📊 Dataset Overview
The dataset includes:

| Feature         | Description                                 |
|----------------|---------------------------------------------|
| `Fan`, `Refrigerator`, `AirConditioner`, `Television`, `Monitor`, `MotorPump` | Appliance counts |
| `MonthlyHours` | Total usage hours per month                 |
| `TariffRate`   | Tariff cost per electricity unit            |
| `City`, `Company` | Categorical features (encoded)             |
| `ElectricityBill` | Target variable — monthly electricity bill |
| `CostPerHour`  | ⚡ *Engineered feature: bill per usage hour* |

---

## 🛠️ Preprocessing & Feature Engineering
- Handled categorical encoding using `LabelEncoder`
- Scaled numerical values using `StandardScaler`
- Engineered a new feature: **`CostPerHour = ElectricityBill / MonthlyHours`**

---

## 🤖 Model
- **Random Forest Regressor** (sklearn)
- 80/20 train-test split
- Evaluated using **RMSE** and **R² Score**

---

## 📈 Results
- **RMSE:** ~5.31
- **R² Score:** ~1.0000
- `MonthlyHours` was the most influential feature (correlation ≈ 0.96)
- `CostPerHour` added interpretability and performance

---

## 📊 Visuals Included
- Correlation Heatmap
- Actual vs Predicted Plot
- Feature Importance Graph
- MonthlyHours vs ElectricityBill Scatter

---

## 🚀 Innovation Highlight
We introduced an engineered feature `CostPerHour`, providing insight into how efficiently electricity is consumed. This improved interpretability and helped the model learn better patterns.

---

## 📂 Project Structure
ElectricityBillPredictor/
├── data/
│   └── electricity_bill_dataset.csv
├── report/
│   ├── README.md
│   └── model_summary.txt
├── video/
│   └── demo_video.mp4 OR link.txt
├── model_dev.py
└── requirements.txt
