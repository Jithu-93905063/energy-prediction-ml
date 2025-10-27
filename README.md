# ⚡ Energy Consumption Prediction Project

This project predicts **building energy consumption** using **machine learning models** trained on weather and building metadata.  
It uses **Random Forest Regressor** for hourly predictions and **LightGBM** for monthly aggregated energy predictions.

---
## 🎯 Objective

The goal is to build an accurate ML model to predict **meter readings** (energy consumption)  
using data such as:
- Building size and usage type
- Weather conditions
- Time-based features (hour, day, month)

---
## 🧠 Models Used

| Model | Purpose | Library | Key Features |
|--------|----------|----------|---------------|
| Random Forest | Hourly prediction | Scikit-learn | Handles non-linear relationships |
| LightGBM | Monthly prediction | LightGBM | Fast, gradient-boosted decision trees |

---
