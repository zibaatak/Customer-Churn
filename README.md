# E-commerce Customer Churn Analysis and Prediction

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Exploratory Data Analysis](#ExploratoryDataAnalysis)
- [Results](#results)
- [Summary](#summary)
- [Acknowledgments](#acknowledgments)

## Overview

Welcome to the E-commerce Customer Churn Analysis and Prediction project! Our objective is to analyze and predict customer churn, 
a critical challenge for businesses striving to retain their valuable clientele. By harnessing the power of data-driven insights,
the aim is to offer strategic suggestions to minimize churn rates and enhance customer retention strategies.

## Data

The dataset, sourced from Kaggle, provides a comprehensive view of customer attributes and behaviors that influence churn within 
the e-commerce domain. Each data point is a window into the dynamics of customer engagement, revealing key aspects that impact 
their loyalty and satisfaction. Here's a breakdown of the dataset's essential components:

E Comm CustomerID: A unique identifier for each customer in the dataset.
E Comm Churn: A churn flag indicating whether a customer has churned or not.
E Comm Tenure: The duration of the customer's association with the e-commerce organization.
E Comm PreferredLoginDevice: The preferred device for customer logins.
E Comm CityTier: The tier of the city where the customer resides.
E Comm WarehouseToHome: The distance between the customer's residence and the e-commerce warehouse.
E Comm PreferredPaymentMode: The customer's preferred method of payment.
E Comm Gender: The gender of the customer.
E Comm HourSpendOnApp: The number of hours the customer spends on the e-commerce mobile application or website.
E Comm NumberOfDeviceRegistered: The total number of devices registered by the customer.
E Comm PreferedOrderCat: The preferred order category of the customer in the last month.
E Comm SatisfactionScore: The customer's satisfaction score related to the service.
E Comm MaritalStatus: The marital status of the customer.
E Comm NumberOfAddress: The total number of addresses associated with the customer.
E Comm Complain: Whether the customer has raised any complaints in the last month.
E Comm OrderAmountHikeFromlastYear: The percentage increase in order amount from the previous year.
E Comm CouponUsed: The total number of coupons used by the customer in the last month.
E Comm OrderCount: The total number of orders placed by the customer in the last month.
E Comm DaySinceLastOrder: The number of days since the customer's last order.
E Comm CashbackAmount: The average cashback received by the customer in the last month.

By dissecting this information, the aim is to uncover patterns, correlations, and predictive indicators that will empower 
businesses to make informed decisions in their pursuit of customer satisfaction and retention.

## Exploratory Data Analysis



## Results
**1. Data Overview**
The dataset comprises 5,630 customer records with 20 columns, including various features and the target variable "Churn," which indicates whether a customer has churned (1 for yes, 0 for no). Here's an overview of the dataset:

Numerical Features: The dataset includes both numerical and categorical features.

Numerical features include "Tenure," "WarehouseToHome," "HourSpendOnApp," "NumberOfDeviceRegistered," "NumberOfAddress," "OrderAmountHikeFromlastYear," "CouponUsed," "OrderCount," "DaySinceLastOrder," and "CashbackAmount."
Categorical features include "PreferredLoginDevice," "CityTier," "PreferredPaymentMode," "Gender," "PreferedOrderCat," "SatisfactionScore," and "MaritalStatus."
Missing Data: Several columns had missing data:

"Tenure" had 264 missing values.
"WarehouseToHome" had 251 missing values.
"HourSpendOnApp" had 255 missing values.
"OrderAmountHikeFromlastYear" had 265 missing values.
"CouponUsed" had 256 missing values.
"OrderCount" had 258 missing values.
"DaySinceLastOrder" had 307 missing values.

**2. Data Preprocessing**
Data Type Conversions:
The "CustomerID" column was converted to an object data type.
Selected columns containing numerical data were converted to integer data types, including "OrderAmountHikeFromlastYear," "CouponUsed," "OrderCount," "NumberOfDeviceRegistered," "NumberOfAddress."
Outlier Handling:
Outliers in numerical features were identified and treated using the IQR method. Values outside 1.5 times the IQR were replaced with the nearest threshold value.
Categorical Data Mapping:
The "PreferredPaymentMode" column was restructured to merge similar payment modes. For instance, "Credit Card," "Debit Card," and "CC" were combined into "Card."

**3. Data Imputation**
**Missing values in numerical columns were imputed using the mean value for each respective column.

**4. Data Cleansing**
**Outliers in numerical columns were detected and corrected using the IQR method to ensure the data's robustness.

**5. Correlation Analysis**
A correlation matrix was computed to examine relationships among numerical features. Notable findings include:

"HourSpendOnApp" had a moderate positive correlation with "CouponUsed" and "OrderCount."
"CouponUsed" exhibited a strong positive correlation with "OrderCount."

**6. Data Transformation and Split**
Feature Engineering:
Categorical features were one-hot encoded using pd.get_dummies() to convert them into numerical form for modeling.
Data Split:
The dataset was split into training and testing sets using the train_test_split function. 80% of the data was allocated for training and 20% for testing.

**7. Standardization** 
Standard Scaling:
Standard scaling was applied to the dataset to ensure that all features had a mean of 0 and a standard deviation of 1. This preprocessing step is essential for certain machine learning algorithms, such as logistic regression and support vector machines, to perform optimally.

**8. Logistic Regression**
Hyperparameter Tuning:
Hyperparameter tuning was performed for logistic regression using GridSearchCV to find the best combination of hyperparameters. The best parameters were found to be {'C': 1.0, 'l1_ratio': 0.0, 'penalty': 'l1', 'solver': 'saga'} with a best score of 0.8839.
Model Evaluation:
The logistic regression model was trained on the scaled training data and evaluated on the test data.
Classification Report:
The classification report showed the following results:
Accuracy: 91%
Precision: 84% for "Yes" (churned) and 92% for "No" (not churned).
Recall: 58% for "Yes" and 98% for "No."
F1-score: 69% for "Yes" and 95% for "No."
Class Distribution:
The class distribution in the dataset revealed that approximately 83.16% of customers did not churn ("No"), while 16.84% churned ("Yes").
Receiver Operating Characteristic (ROC) Curve and Area Under Curve (AUC):
The ROC curve for logistic regression had an AUC of 0.90, indicating good discriminative power.
Average Precision (AP):
The Average Precision (AP) for logistic regression was 0.75, measuring the precision-recall trade-off.

**9. Random Forest**
Hyperparameter Tuning:
Hyperparameter tuning was performed for random forest using GridSearchCV. The best parameters were found to be {'bootstrap': False, 'max_depth': None, 'max_features': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'oob_score': False}, with a best score of 0.9603.
Model Evaluation:
The random forest model was trained with the best parameters on the scaled training data and evaluated on the test data.
Classification Report:
The classification report showed the following results:
Accuracy: 98%
Precision: 100% for "Yes" and 98% for "No."
Recall: 90% for "Yes" and 100% for "No."
F1-score: 95% for "Yes" and 99% for "No."
Receiver Operating Characteristic (ROC) Curve and Area Under Curve (AUC):
The ROC curve for random forest had an AUC of 0.99, indicating excellent discriminative power.
Average Precision (AP):
The Average Precision (AP) for random forest was 0.9795, indicating high precision in classifying positive instances.

**10. Gradient Boosting**
Hyperparameter Tuning:
Hyperparameter tuning was performed for gradient boosting using GridSearchCV. The best parameters were {'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 300}, with a best score of 0.9636.
Model Evaluation:
The gradient boosting model was trained with the best parameters on the scaled training data and evaluated on the test data.
Classification Report:
The classification report showed the following results:
Accuracy: 98%
Precision: 97% for "Yes" and 98% for "No."
Recall: 91% for "Yes" and 99% for "No."
F1-score: 94% for "Yes" and 99% for "No."

**11. Support Vector Machine (SVM)**
Hyperparameter Tuning:
Hyperparameter tuning was performed for SVM using GridSearchCV. The best parameters were {'C': 70, 'gamma': 'scale', 'kernel': 'rbf'}, with a best score of 0.9518.
Model Evaluation:
The SVM model was trained with the best parameters on the scaled training data and evaluated on the test data.
Classification Report:
The classification report showed the following results:
Accuracy: 98%
Precision: 95% for "Yes" and 98% for "No."
Recall: 92% for "Yes" and 99% for "No."
F1-score: 93% for "Yes" and 99% for "No."

**##Summary**
The objective of this churn prediction project was to develop and evaluate machine learning models capable of identifying customers at risk of churning. The dataset underwent several preprocessing steps, including feature engineering, standardization, and data splitting, to prepare it for model training and evaluation.

**Model Performance**
**- Logistic Regression**
Logistic regression, a simple yet interpretable model, achieved an accuracy of 91%. While it showed relatively high precision for "No" (not churned) customers at 92%, its precision for "Yes" (churned) customers was 84%. This discrepancy in precision might indicate that the model is slightly biased towards predicting non-churners. The ROC AUC score was 0.90, indicating good discriminative power. However, the Average Precision (AP) was 0.75, suggesting room for improvement in precision-recall balance.

**- Random Forest**
Random forest, a more complex ensemble model, demonstrated impressive performance with an accuracy of 98%. It showed high precision for both "Yes" (100%) and "No" (98%) classes, indicating a balanced prediction. The ROC AUC score was excellent at 0.99, indicating exceptional discriminative power. The high Average Precision (AP) of 0.9795 underscores the model's precision in identifying churners. However, monitoring for potential overfitting is important due to the model's complexity.

**- Gradient Boosting**
Gradient boosting, known for its ability to handle complex relationships, delivered an accuracy of 98%. The model achieved 97% precision for "Yes" and 98% for "No," indicating a robust balance. The ROC AUC score of 0.99 signifies superior discriminative power. The AP score of 0.9636 highlights strong precision in classifying churners. Care should be taken to monitor the model's complexity and potential overfitting.

**- Support Vector Machine (SVM)**
SVM, with tuned hyperparameters, achieved an accuracy of 98%. It demonstrated 95% precision for "Yes" and 98% for "No," maintaining a balanced prediction. The ROC AUC score of 0.99 indicates excellent discriminative power. The AP score of 0.9518 demonstrates good precision in classifying churners.

**Model Evaluation and Considerations**
**Overfitting:** While the models showed high accuracy and performance, it's essential to remain vigilant for signs of overfitting, especially in more complex models like random forest and gradient boosting. Regular monitoring and validation on unseen data are crucial.

**ROC AUC:** The high ROC AUC scores across all models suggest their effectiveness in distinguishing between churners and non-churners. A higher AUC indicates better overall performance.

**Average Precision (AP):** The Average Precision metric, especially high in the random forest model, emphasizes precision in identifying churners. This is essential when the cost of false positives is high, as it ensures that customers identified as at risk of churning are indeed likely to do so.

In conclusion, this project successfully built and evaluated machine learning models for churn prediction. The choice of the final model should consider the business context, including the cost of false positives and false negatives. Regular model monitoring and maintenance are essential to ensure continued accuracy and relevance.


## Acknowledgments

Thank you to the Kaggle community for providing the E-commerce Customer Churn dataset.  https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

