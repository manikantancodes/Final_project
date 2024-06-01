# Machine Learning Projects

This repository houses three separate machine learning projects that tackle various predictive analytics challenges. The projects include breast cancer prediction using ensemble techniques, e-commerce customer segmentation via k-means clustering, and term deposit subscription prediction with support vector machines (SVM). Each project involves steps like data preprocessing, exploratory data analysis (EDA), model development, evaluation, and hyperparameter tuning.


## Projects

1. [Predicting Breast Cancer in a Patient](#predicting-breast-cancer-in-a-patient)
2. [E-commerce Customer Segmentation](#e-commerce-customer-segmentation)
3. [Predicting Term Deposit Subscription by a Client](#predicting-term-deposit-subscription-by-a-client)



## Project 1: Predicting Breast Cancer in Patients

### Overview
Breast cancer is a significant cause of mortality among women globally. This project aims to classify whether a patient has breast cancer using ensemble methods. The dataset includes numerous features derived from cell nuclei measurements.

### Data Details
The dataset includes predictor variables that describe characteristics of cell nuclei, with a target variable `Diagnosis` that indicates 'Benign' (no cancer) or 'Malignant' (cancer).

### Project Workflow:
1. **Data Preprocessing & Exploratory Data Analysis (EDA)**
2. **Model Building with Ensemble Techniques**
3. **Model Evaluation**
4. **Hyperparameter Optimization**

### Main Script:
- `breast_cancer_prediction.ipynb`: Contains code for preprocessing, EDA, model training, evaluation, and tuning.

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC Score

### Execution:
`breast_cancer_prediction.ipynb`


# Project 2: E-commerce Customer Segmentation

## Overview
This project focuses on clustering e-commerce customers based on their activity using the k-means clustering algorithm. Customer segmentation is essential for understanding customer behavior and improving targeting strategies.

## Data Details
The dataset includes customer IDs, gender, order counts, and search frequencies for various brands.

### Variable Description:
- **Cust_ID**: Unique identifier for customers
- **Gender**: Gender of the customer
- **Orders**: Number of orders placed by each customer in the past
- Remaining 35 features represent the number of times customers have searched specific brands

## Project Workflow
1. **Data Preprocessing & Exploratory Data Analysis (EDA)**
   - Handle missing values
   - Analyze the distribution of variables
   - Visualize correlations between features

2. **Model Building with K-means Clustering**
   - Standardize the data
   - Apply Principal Component Analysis (PCA) for dimensionality reduction
   - Determine the optimal number of clusters using the silhouette score
   - Apply k-means clustering to segment customers

3. **Evaluation & Visualization**
   - Analyze cluster characteristics
   - Visualize cluster distributions

## Data Preprocessing
- Missing values are handled by imputing or dropping missing data
- Standardization of features is done to ensure equal importance of all features

## Model Building
- **Principal Component Analysis (PCA)**: Reduces dimensionality to make clustering more efficient and interpretable
- **K-means Clustering**: Segments customers into clusters based on their search and purchase behavior
- **Silhouette Score**: Helps determine the optimal number of clusters

## Scripts
- `customer_segmentation.ipynb`: Contains code for preprocessing, EDA, model training, and evaluation.

## Evaluation Metric
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters

## Running the Project
To run the customer segmentation project, execute the following command in your terminal:
`customer_segmentation.ipynb`


# Project 3: Predicting Term Deposit Subscription by a Client

## Overview
This project aims to predict whether a client will subscribe to a term deposit when contacted by a marketing agent. By analyzing various client attributes and previous campaign data, we use Support Vector Machine (SVM) for the predictive analytics.

## Data Details
The dataset contains information related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls, and multiple contacts were often made to the same client.

### Variable Description:
- **age**: Age of the client
- **job**: Type of job (categorical)
- **marital**: Marital status (categorical)
- **education**: Education level (categorical)
- **default**: Credit in default? (categorical)
- **housing**: Has a housing loan? (categorical)
- **loan**: Has a personal loan? (categorical)
- **contact**: Contact communication type (categorical)
- **month**: Last contact month of the year (categorical)
- **day_of_week**: Last contact day of the week (categorical)
- **duration**: Last contact duration, in seconds
- **campaign**: Number of contacts performed during this campaign and for this client
- **pdays**: Number of days since the client was last contacted from a previous campaign
- **previous**: Number of contacts performed before this campaign for this client
- **poutcome**: Outcome of the previous marketing campaign (categorical)
- **emp.var.rate**: Employment variation rate - quarterly indicator
- **cons.price.idx**: Consumer price index - monthly indicator
- **cons.conf.idx**: Consumer confidence index - monthly indicator
- **euribor3m**: Euribor 3 month rate - daily indicator
- **nr.employed**: Number of employees - quarterly indicator
- **y**: Has the client subscribed a term deposit? (binary: 'yes','no')

## Project Workflow
1. **Data Preprocessing & Exploratory Data Analysis (EDA)**
   - Handle missing values
   - Encode categorical variables
   - Analyze the distribution of variables
   - Split the data into training and testing sets

2. **Model Building with Support Vector Machine (SVM)**
   - Train the SVM classifier
   - Make predictions on the testing set
   - Evaluate the model using various metrics

3. **Hyperparameter Tuning**
   - Tune the hyperparameters of the SVM model to improve performance

4. **Evaluation & Visualization**
   - Assess model performance using metrics like precision, recall, F1-score, and AUC-ROC
   - Visualize confusion matrix and ROC curve

## Data Preprocessing
- Encode categorical variables using Label Encoding
- Handle missing values by imputing or dropping missing data
- Split the data into features (X) and target variable (y)
- Perform train-test split with 80% training data and 20% testing data

## Model Building
- **Support Vector Machine (SVM)**: A robust classifier used for prediction
- **Hyperparameter Tuning**: Use techniques like grid search or random search to find the best parameters for the SVM model

## Scripts
- `term_deposit_prediction.ipynb`: Contains code for preprocessing, EDA, model training, hyperparameter tuning, and evaluation.

## Evaluation Metrics
- **Accuracy**: Measures the overall correctness of the model
- **Precision**: Measures the correctness of positive predictions
- **Recall (Sensitivity)**: Measures the ability to identify positive instances
- **Specificity**: Measures the ability to identify negative instances
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Measures the area under the receiver operating characteristic curve

## Running the Project
Open and run the 
`term_deposit_prediction.ipynb` notebook.
