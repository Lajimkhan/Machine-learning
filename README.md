# 🌍 **Landslide Susceptibility Prediction Using Machine Learning**

A machine learning–based system for predicting **landslide occurrence** using environmental, geological, and climatic features.
The project applies **Gradient Boosting**, advanced **data preprocessing**, class rebalancing with **RandomUnderSampler**, and multiple visualization techniques to evaluate model performance.

This repository is ideal for:

✔ Environmental scientists
✔ Machine learning researchers
✔ GIS analysts
✔ Students working on natural hazard prediction

---

## 📌 **Key Features**

* **Full ML Pipeline**: From loading data → preprocessing → modeling → evaluation

* **Class Imbalance Handling** using RandomUnderSampler

* **Gradient Boosting Classifier** for robust tabular prediction

* **Rich Visual Analysis** including:

  * ROC Curve
  * Precision–Recall Curve
  * F1–Threshold Curve
  * Accuracy–Threshold Curve
  * Learning Curve
  * Confusion Matrix
  * Correlation Heatmap

* **High Accuracy** (99%+ on test data)

---

## 📂 **Project Structure**

```
📁 Landslide-Prediction-ML
│
├── dataset/
│   └── new1SupervisedDataSet.csv
│
├── src/
│   └── landslide_prediction.ipynb
│
├── README.md
└── requirements.txt
```

---

## ⚙️ **Technologies Used**

* Python 3
* Pandas, NumPy
* Scikit-Learn
* Imbalanced-Learn
* Matplotlib, Seaborn
* Google Colab (optional)

---

## 🚀 **How to Run the Project**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Lajimkhan/Landslide-Prediction-ML.git
cd Landslide-Prediction-ML
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Open the Notebook

```bash
jupyter notebook
```

Run:

```
src/landslide_prediction.ipynb
```

---

# 📊 **Dataset Overview**

The dataset includes 17 important features influencing landslides, such as:

* Elevation
* Slope
* Curvature
* Lithology
* NDVI / NDWI
* Rainfall
* Humidity
* Moisture
* Pressure
* Earthquake activity

These features contribute significantly to understanding slope instability.

---

# 🧠 **Model: Gradient Boosting Classifier**

Gradient Boosting is effective for:

✔ Handling nonlinear relationships
✔ High accuracy with minimal tuning
✔ Strong performance on environmental datasets

The model was trained on **undersampled** data to address imbalance.

---

# 📈 **Model Performance**

| Metric        | Score  |
| ------------- | ------ |
| **Accuracy**  | 0.997  |
| **Precision** | 0.985  |
| **Recall**    | 0.958  |
| **F1 Score**  | 0.971  |
| **R² Score**  | 0.960  |
| **MSE**       | 0.0095 |

**Conclusion:**
➡ The model performs exceptionally well and is stable across evaluation metrics.

---

# 🖼️ **Visualizations Included**

The notebook produces the following visuals:

### 🔹 **1. Class Distribution (Before & After Resampling)**

Shows how undersampling balances the dataset.

### 🔹 **2. Correlation Heatmap**

Reveals feature relationships.

### 🔹 **3. Confusion Matrix**

Displays classification performance.

### 🔹 **4. Precision–Recall Curve**

### 🔹 **5. ROC Curve & AUC Score**

### 🔹 **6. F1 Score vs Threshold**

### 🔹 **7. Accuracy vs Threshold**

### 🔹 **8. Learning Curve**

Shows bias–variance characteristics.

---

# 🔮 **Future Improvements**

To make this project even more powerful, consider the following upgrades:

### ⭐ **1. Test More ML Models**

* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Logistic Regression (baseline)

### ⭐ **2. Use Hyperparameter Optimization**

* GridSearchCV
* RandomizedSearchCV
* **Optuna** (best for complex models)

### ⭐ **3. Add Feature Selection & Explainability**

* SHAP values
* LIME
* Permutation importance

This helps environmental experts understand *why* the model predicts landslides.

### ⭐ **4. Try Different Resampling Techniques**

* SMOTE
* ADASYN
* SMOTEENN

Better for highly imbalanced data.

### ⭐ **5. Deploy as a Web App**

* Build a **Streamlit** or **Flask** dashboard
* Enable real-time prediction

### ⭐ **6. GIS Integration**

* Combine with QGIS / ArcGIS
* Produce landslide susceptibility maps

### ⭐ **7. AutoML Pipeline**

Create a fully automated training pipeline with:

* Feature scaling
* Feature selection
* Resampling
* Model comparison
* Automatic reporting

---

# 🏆 **Conclusion**

This project demonstrates a strong machine learning approach for landslide susceptibility prediction, achieving high accuracy and offering a complete evaluation pipeline.
It is an excellent foundation for research, environmental analysis, and real-world deployment.

---

