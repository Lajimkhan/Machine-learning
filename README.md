Landslide Prediction Using Gradient Boosting & Data Resampling
🛰️ Landslide Susceptibility Prediction (Machine Learning Project)

This project focuses on predicting landslide occurrence using a supervised machine learning pipeline.
It includes:

Data preprocessing

Handling class imbalance with RandomUnderSampler

Gradient Boosting classification

Evaluation using multiple metrics

Visualization of model performance (confusion matrix, ROC, PR curve, F1 curve, learning curve)

The dataset contains environmental, geographical, and climatic features such as Elevation, Slope, NDVI, Lithology, Rainfall, Humidity, Pressure, etc.

📂 Project Structure
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

🔧 Technologies & Libraries Used

Python 3.x

Pandas, NumPy

Scikit-Learn

Imbalanced-learn (RandomUnderSampler)

Matplotlib, Seaborn

Google Colab (optional)

📥 How to Run the Project
1. Clone the Repository
git clone https://github.com/yourusername/Landslide-Prediction-ML.git
cd Landslide-Prediction-ML

2. Install Required Libraries

Create a requirements.txt file or use:

pip install -r requirements.txt


Or manually:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

3. Open the Notebook

Use Jupyter/Colab:

jupyter notebook


Run:

src/landslide_prediction.ipynb

🧠 Model Used: Gradient Boosting Classifier

Gradient Boosting is used due to its strong performance on tabular data and ability to model complex interactions.

📊 Evaluation Metrics

The project evaluates the model using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

ROC Curve & AUC

Precision–Recall Curve

Learning Curve

Error Metrics (R² Score, MSE)

Model Performance
Metric	Score
Accuracy	0.99+
Precision	0.98
Recall	0.95
F1 Score	0.97
R² Score	0.96
MSE	0.0095
🖼️ Visualizations Included

Class distribution before/after resampling

Correlation heatmap

Confusion matrix

ROC curve

Precision–Recall curve

F1 score vs. threshold

Accuracy vs. threshold

Learning curve

🚀 Future Improvements

Here are recommended improvements to enhance your project:

🔹 1. Try More Models

Experiment with:

Random Forest

XGBoost

LightGBM

CatBoost

SVM

Logistic Regression (baseline)

🔹 2. Hyperparameter Tuning

Use:

GridSearchCV
RandomizedSearchCV
Optuna (best option)

🔹 3. Feature Engineering

Create composite features

Conduct feature selection (SHAP, permutation importance)

Handle outliers

🔹 4. Use Advanced Sampling Techniques

Instead of RandomUnderSampler:

SMOTE

ADASYN

SMOTEENN (hybrid)

🔹 5. Deploy the Model

Deploy using:

Flask / FastAPI

Streamlit interactive dashboard

Docker container

Google Cloud / AWS deployment

🔹 6. Integrate GIS

Combine ML with:

QGIS

ArcGIS

DEM generation

Geospatial heatmaps

🔹 7. Real-Time Monitoring

Enable periodic retraining using:

Weather updates

Rainfall predictions

Satellite imagery (NDVI/NDWI updates)

🔹 8. Add Explainability (XAI)

Use:

SHAP values

LIME
To understand how each feature affects landslide prediction.

🏁 Conclusion

This project demonstrates a full ML pipeline for landslide susceptibility prediction using a Gradient Boosting Classifier.
The current model achieves excellent accuracy and performs well across multiple metrics.
