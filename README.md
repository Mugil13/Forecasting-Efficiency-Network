# Forecasting Efficiency Network

This repository implements an interesting methodology that combines multiple ML and DL models and **Operations Research (DEA - Malmquist Index)** to forecast and understand the trends of sugar-production quantities of countries. The system integrates various machine learning regressors, including GRU, LSTM, Random Forest, SVR and other ensemble models as well, followed by Data Envelopment Analysis on predicted values to evaluate future **Total Factor Productivity (TFP)**.

The methodology is inspired by the hybrid DEA-RNN model proposed in:
> Zhang et al. (2024), “A hybrid approach combining data envelopment analysis and recurrent neural network for predicting the efficiency of research institutions”, *Expert Systems with Applications*, Elsevier.  
> [[DOI:10.1016/j.eswa.2023.122150]](https://doi.org/10.1016/j.eswa.2023.122150)

---

## Features

- Time-Series Forecasting using **GRU**, **LSTM**, and other regressors.
- Efficiency evaluation using **DEA Malmquist Index**.
- Performance metrics: RMSE, MAPE, Accuracy.
- Multi-model comparison across 8 different countries

---

## Preprocessing

- Data Normalization
- Sliding Window Time-Series Conversion
- One-Step Forecasting Setup

---

## Models & Parameters used

| Model              | Key Hyperparameters |
|-------------------|---------------------|
| Random Forest      | `n_estimators=100`, `random_state=42` |
| Extra Trees        | `n_estimators=100`, `random_state=42` |
| SVR                | `kernel='rbf'`, `C=100`, `epsilon=0.1` |
| LASSO              | `alpha=0.1` |
| ElasticNet         | `alpha=1.0`, `l1_ratio=0.5` |
| XGBoost            | `n_estimators=100`, `random_state=42` |
| Bagging Regressor  | `base_estimator=DecisionTreeRegressor`, `n_estimators=100` |
| **LSTM**           | `epochs=10`, `optimizer='adam'`, `early stopping` |
| **GRU**            | Same as LSTM |
| IOMN Regularized   | `hidden=32`, `dropout=0.3`, `l2=1e-4`, `epochs=10` |

---

## Evaluation & Results

Each model is evaluated using:

- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error  
- **Accuracy Rate (AC)**: 1 - MAPE

| Country      | Top Performer(s)       | 
|--------------|------------------------|
| **Australia** | Random Forest, XGBoost |
| **Brazil**    | XGBoost, Blending      |
| **Colombia**  | Bagging, Blending      |
| **Guatemala** | ElasticNet, RF         |
| **India**     | ElasticNet, RF         |
| **Mexico**    | ElasticNet, Blending   |
| **S. Africa** | LASSO, Bagging         |
| **Thailand**  | ElasticNet, RF         |

*In most countries, tree-based models (RF, XGBoost, Bagging) outperform deep learning models in MAPE and Accuracy.*

---

## How to Run
```
git clone https://github.com/Mugil13/Forecasting-Efficiency-Network.git
cd Forecasting-Efficiency-Network
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgment
Special thanks to Zhang et al. (2024) for the foundational hybrid methodology and DEA insights used in this implementation.
