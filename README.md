# 📈 Demand Forecasting App

A machine learning project that analyzes historical retail demand data, engineers meaningful features, trains an XGBoost regression model to predict product demand, and serves live predictions through an interactive Streamlit web app.

---

## 🔍 Overview

This project walks through a full data science pipeline:

1. **Exploratory Data Analysis (EDA)** — understanding demand patterns across categories, regions, seasons, weather, and promotions.
2. **Feature Engineering** — extracting date-based features and calculating discounted pricing.
3. **Model Training** — tuning an XGBoost regressor with `RandomizedSearchCV` to predict demand.
4. **Deployment** — a Streamlit app where users can input feature values and get a live demand prediction.

---

## 📊 Dataset

The dataset (`demand_forecasting.csv`) contains historical records with fields including:

- `Store ID`, `Product ID`, `Category`, `Region`
- `Date`, `Weather Condition`, `Seasonality`
- `Price`, `Discount`, `Competitor Pricing`
- `Inventory Level`, `Units Sold`, `Promotion`, `Epidemic`
- `Demand` (target variable)

---

## 🛠️ Feature Engineering

- Extracted `Year`, `Month`, `Day`, and `Weekday` from the `Date` column
- Calculated `Discounted Price` from `Price` and `Discount`
- Encoded the categorical `Category` column using `LabelEncoder`

---

## 📈 Exploratory Analysis Highlights

- Compared average demand across `Category`, `Region`, `Seasonality`, and `Weather Condition`
- Analyzed the impact of `Promotion` on demand (demand rises noticeably during active promotions)
- Visualized demand trends over time (daily and monthly aggregations)
- Examined the relationship between `Discounted Price` and `Demand`

---

## 🤖 Model

- **Algorithm:** XGBoost Regressor (`XGBRegressor`)
- **Hyperparameter tuning:** `RandomizedSearchCV` (25 iterations, 3-fold CV, scored on negative MAE)
- **Best parameters found:**
  ```python
  {
      'subsample': 0.8,
      'n_estimators': 300,
      'min_child_weight': 5,
      'max_depth': 8,
      'learning_rate': 0.05,
      'colsample_bytree': 1.0
  }
  ```
- **Evaluation metrics:** MAE, RMSE, R²
- **Feature importance:** `Promotion` emerged as the strongest predictor of demand, followed by `Category` and `Price`.

---

## 🚀 Streamlit App

The app (`app.py`) lets users:

- Input values for Price, Discount, Inventory Level, Promotion, Competitor Pricing, and Category
- Get an instant predicted demand value from the trained model

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                              # Streamlit app
├── analysis.ipynb                      # EDA notebook
├── machine_learning.ipynb              # Model training notebook
├── demand_forecasting.csv              # Raw dataset
├── preprocessed_demand_forecasting_data.csv   # Cleaned & feature-engineered dataset
├── xgboost_demand_model.pkl            # Trained model
├── label_encoders.pkl                  # Saved label encoders
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🧰 Tech Stack

- **Language:** Python
- **Data analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, XGBoost
- **App/Deployment:** Streamlit

---

## 📌 Future Improvements

- Add time-series-specific models (e.g., ARIMA, Prophet) for comparison
- Incorporate confidence intervals on predictions
- Add batch prediction support (CSV upload) in the app
- Deploy with CI/CD for automatic redeployment on code changes

---

## 📄 License

This project is open-source and available for personal or educational use.
