# 🚧 Landslide Prediction Using Gradient Boosting (Machine Learning Project)

This repository contains a complete machine learning workflow for **landslide susceptibility prediction** using environmental and geological features. The project includes **data preprocessing**, **class balancing using RandomUnderSampler**, **model training**, and **detailed evaluation** with visualizations.

---

## 📌 **Project Overview**

Landslides are major natural hazards influenced by various environmental and climatic factors.
This machine learning model predicts landslide occurrence (0 or 1) using features such as:

* Elevation
* Slope
* Curvature
* Lithology
* NDVI & NDWI
* Rainfall
* Temperature
* Humidity
* Earthquake data
* Moisture & Pressure
* And many other physical attributes

The dataset has **190,890 samples** and **17 feature columns**.

Machine learning model used:
🔹 **Gradient Boosting Classifier (GBC)**

---

## 📁 **Dataset**

The dataset is stored in Google Drive and loaded into Google Colab:

```
/content/drive/MyDrive/ml Final/new1SupervisedDataSet.csv
```

### **Target Variable**

* **Landslide (0 or 1)**

### **Features**

`Aspect, Curvature, Earthquake, Elevation, Flow, Lithology, NDVI, NDWI, Plan, Precipitation, Profile, Slope, temperature, humidity, rain, moisture, pressure`

---

## ⚙️ **Tech Stack / Libraries Used**

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* scikit-learn
* imbalanced-learn
* Google Colab

---

## 🧹 **Data Processing Steps**

1. **Load dataset**
2. **Check missing values**
3. **Separate features & target**
4. **Handle class imbalance using RandomUnderSampler**
5. **Split into training and testing sets**
6. **Train Gradient Boosting Classifier**
7. **Evaluate model performance**

---

## 📊 **Model Evaluation Metrics**

| Metric                               | Score     |
| ------------------------------------ | --------- |
| **Accuracy (Undersampled Test Set)** | `0.9904`  |
| **R2 Score**                         | `0.9604`  |
| **MSE**                              | `0.00959` |
| **Final Test Accuracy**              | `0.9971`  |
| **Precision**                        | `0.9849`  |
| **Recall**                           | `0.9582`  |
| **F1-score**                         | `0.9714`  |

---

## 📈 **Visualizations Included**

✔️ Original Class Distribution
✔️ Undersampled Class Distribution
✔️ Correlation Heatmap
✔️ Confusion Matrix
✔️ Precision–Recall Curve
✔️ F1-score Curve
✔️ ROC Curve
✔️ Accuracy vs Threshold Curve
✔️ Learning Curve (Train vs Cross-validation)

---

## 📦 **How to Run the Code**

### **1. Clone this repository**

```bash
git clone https://github.com/yourusername/your-repository-name.git
```

### **2. Open Google Colab**

Upload the notebook or copy the script.

### **3. Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

### **4. Install Dependencies**

```bash
pip install pandas numpy seaborn scikit-learn imbalanced-learn matplotlib
```

### **5. Run the script**

The script handles:

* Data loading
* Preprocessing
* Undersampling
* Model training
* Evaluation
* Plotting

---

## 🚀 **Future Improvements**

* Use SMOTE + Tomek Links for better balancing
* Try XGBoost / LightGBM for higher accuracy
* Hyperparameter tuning with GridSearchCV / Optuna
* Deployment using Flask or FastAPI
* Convert model to ONNX
* Add SHAP explainability

---
