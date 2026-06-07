# P2P Loan Risk Prediction System

This project is an end-to-end machine learning solution designed to predict the likelihood of loan default in Peer-to-Peer (P2P) lending environments. Using historical Lending Club loan data, the system analyzes borrower characteristics and predicts whether a loan is likely to be fully paid or charged off.

The goal is to provide borrowers and lenders with a transparent decision-support tool that offers both accurate risk predictions and interpretable explanations of model decisions.

## Features

* End-to-end machine learning pipeline
* Data preprocessing and feature engineering
* Comparative evaluation of multiple classification algorithms
* Interactive web-based prediction interface
* Real-time loan risk assessment
* Model interpretability through local sensitivity analysis
* Visualization of factors contributing to increased or decreased risk

## Dataset

The project uses publicly available Lending Club loan data containing borrower, loan, and repayment information. Historical loan outcomes are used to train supervised learning models to distinguish between fully paid and charged-off loans.

## Machine Learning Models

The following classification models were implemented and evaluated:

1. Logistic Regression (SGD-based)
2. Random Forest
3. k-Nearest Neighbors (k-NN)
4. XGBoost

Models were trained, tuned, and compared using multiple performance metrics.

## Evaluation Metrics

Model performance was assessed using:

* AUROC
* F1-Score
* Accuracy
* Precision

After comparative evaluation, XGBoost achieved the strongest overall performance and was selected for deployment.

## System Architecture

1. Data Acquisition
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Model Training and Hyperparameter Tuning
5. Model Evaluation
6. Model Deployment
7. Prediction and Explainability Interface

## Web Application

The deployed application allows users to:

* Enter borrower and loan information
* Receive an instant risk prediction
* View the estimated likelihood of default
* Understand the factors influencing the prediction

## Explainability

To improve transparency, the application includes a local sensitivity analysis module that highlights the borrower attributes with the greatest positive and negative impact on risk. This helps users understand how individual features influence the model's prediction.

## Technologies Used

### Machine Learning

* Python
* Scikit-learn
* XGBoost
* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Web Application

* Flask (or applicable framework)
* HTML
* CSS
* JavaScript

## Results

Among the evaluated models, XGBoost provided the best balance of predictive performance and robustness, making it the final model selected for deployment. Combined with explainability features, the system delivers a practical and transparent solution for loan risk assessment in P2P lending scenarios.

## Future Improvements

* Incorporate additional lending datasets
* Explore deep learning approaches
* Add probability calibration for improved confidence estimates
* Deploy as a cloud-hosted application
* Implement advanced explainability techniques such as SHAP
