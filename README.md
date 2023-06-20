# HDSC-Spring-23-Premier-Project-plotly-group


## Executive Summary
This data analyst report provides an analysis of the Chronic Kidney Disease (CKD) dataset and aims to predict the chances of being diagnosed with chronic kidney disease. The analysis includes data exploration, cleaning, preprocessing, model selection, training, evaluation, and model deployment. The report highlights key findings and recommendations based on the analysis.

## Introduction
Chronic kidney disease is a significant health problem with potentially fatal consequences. This project aims to predict the likelihood of individuals being diagnosed with chronic kidney disease using a provided dataset. By leveraging machine learning algorithms, we can contribute to addressing this health issue and potentially improve patient outcomes.

## Dataset Overview
The CKD dataset consists of a wide range of medical attributes, including age, blood pressure, serum creatinine, glucose, and other relevant features. The dataset comprises both numerical and categorical variables, which are utilized for training and evaluating the predictive models.

Key Findings
- Dataset Structure: The dataset contains 400 rows and [number of 25 columns. 
Each row in the dataset represents a patient, and the columns correspond to different features or attributes of the patients. Here's a breakdown of the columns mentioned:
  - `age`: The age of the patient (in years)
  - `bp`: Blood pressure of the patient (measured in mmHg)
  - `sg`: Specific gravity of urine
  - `al`: Albumin present in the urine
  - `su`: Sugar present in the urine
  - `rbc`: Red blood cells in urine (either "normal" or "abnormal")
  - `pc`: Pus cell in urine (either "normal" or "abnormal")
  - `pcc`: Pus cell clumps in urine (either "present" or "not present")
  - `ba`: Bacteria present in urine (either "present" or "not present")
  - `bgr`: Blood glucose random (measured in mg/dL)
  - `pcv`: Packed cell volume
  - `wbcc`: White blood cell count (measured in cells/cubic mm)
  - `rbcc`: Red blood cell count (measured in millions/cubic mm)
  - `htn`: Whether the patient has hypertension (either "yes" or "no")
  - `dm`: Whether the patient has diabetes mellitus (either "yes" or "no")
  - `cad`: Whether the patient has coronary artery disease (either "yes" or "no")
  - `appet`: Patient's appetite (either "good" or "poor")
  - `pe`: Presence of pedal edema (either "yes" or "no")
  - `ane`: Presence of anemia (either "yes" or "no")
  - `class`: The target variable, indicating the presence or absence of chronic kidney disease (either "ckd" or "notckd")

- Target Variable Distribution: The countplot reveals that the dataset is imbalanced, with a higher proportion of individuals with chronic kidney disease compared to those with the disease. 250 of those with chronic kidney disease and 150 of those without chronic disease.
- Outliers: Outliers were identified in only wbcc (white blood cells count) variable.
- Correlation: The heatmap visualization indicated strong correlations between certain feature pairs, which should be considered during model training.
- Age Distribution: The histogram of age showed that the majority of individuals in the dataset fall within a 40-60 years.
- Blood Pressure vs. Glucose: The scatter plot between blood pressure and glucose random measurement did not reveal any distinct patterns or associations, indicating that there is no significant linear relationship between these two variables.
- Serum Creatinine by Hypertension: Individuals with hypertension tend to have higher serum creatinine levels, as shown in the box plot.
- Pairwise Relationships: The pairplot highlighted potential correlations and patterns between multiple variables, contributing to feature selection and engineering.
- Age Distribution across Classes: The violinplot demonstrated variations in the age distribution across different classes, providing insights into age-related differences.
- Average Hemoglobin Level: The barplot illustrated differences in the average hemoglobin levels between classes, indicating potential significance in predicting chronic kidney disease.



## Library Importation
In this section, we import the necessary libraries and packages required for data analysis, modeling, and evaluation. The common libraries used in this project include:

- Pandas for data manipulation and analysis
- NumPy for numerical operations
- Matplotlib and Seaborn for data visualization
- Scikit-learn for machine learning algorithms and evaluation metrics

## Loading the Dataset
The dataset for this project is available on an [AWS s3 bucket](https://github-data-bucket.s3.amazonaws.com/chronic_kidney_disease.csv) provided by the data engineering team after it has been cleaned and transformed. We load the dataset into a Pandas DataFrame using appropriate functions or methods.

## EDA (Exploratory Data Analysis)
The EDA phase is crucial for understanding the dataset, identifying patterns, and gaining insights. In this section, we perform various exploratory data analysis tasks, including:

- Exploring the structure of the dataset (number of rows, columns, data types, etc.)
- Checking for missing values in the data
- Visualizing the target variable distribution using a countplot
- Checking for outliers using a boxplot
- Examining the statistical summary of the dataset
- Visualizing the correlation of the data using a heatmap
- Visualizing the distribution of age using a histplot
- Visualizing blood pressure (bp) vs. glucose random measurement (bgr) using a scatter plot
- Visualizing serum creatinine (sc) by the presence of hypertension (htn) using a box plot
- Visualizing pairwise relationships between multiple variables, highlighting different classes using a pairplot
- Visualizing the distribution of a continuous variable (age) across different classes using a violinplot
- Visualizing the average hemoglobin level for each class, along with error bars representing the standard deviation using a barplot

## Data Cleaning and Preprocessing
Based on the findings from the EDA, we perform necessary data cleaning and preprocessing steps to ensure the quality and integrity of the data. The steps involved in this section may include:

- Encoding categorical variables and transforming numerical variables
- Performing feature engineering to derive additional features from existing ones
- Separating the features (X) and the target variable (y)
- Balancing the target data using random undersampling
- Creating a bar chart to visualize whether the target is balanced
- Splitting the data into training and testing sets for model training and evaluation

## Model Selection, Training, and Evaluation
In this section, we choose a suitable machine learning model for the given problem and train it using the training dataset. The steps involved are as follows:

- Choosing a Random Forest classification model based on the given data
- Fitting the model to the training data
- Making predictions on the test data
- Evaluating the model's performance on the training and testing datasets using appropriate evaluation metrics
- Iterating this process on four other classification models (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, SVC)
- Choosing the model with the best accuracy


## Model Accuracy Analysis:
When evaluating the performance of the different classification models for predicting chronic kidney disease, we considered accuracy_score and classification metrics: precision, recall, F1 score, and support. The following models were evaluated:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier
- Support Vector Classifier

Among these models, the Random Forest Classifier achieved the best accuracy with a score of 1.0. Let's examine the classification metrics for the best model:
- Best Model: Random Forest Classifier
- Best Accuracy: 1.0

The Random Forest Classifier demonstrated exceptional performance with a perfect accuracy score of 1.0 for predicting chronic kidney disease. 

## Saving the Model for Deployment
After training and evaluating the model, we save it in a serialized format for deployment. The serialized model can be easily loaded and used for making predictions in a production environment. The steps involved in saving the model are:

- Serializing the trained model using Python's pickle library
- Saving the serialized model to a specified file or storage location

## Limitations and Assumptions
The analysis is based on the available dataset and may not capture all potential factors influencing chronic kidney disease prediction.
The assumptions made during the analysis include the assumption that the dataset is representative of the target population and that the selected features are accurate and reliable indicators of the disease.

## Conclusion
This data analyst report provides insights into the prediction of chronic kidney disease using the available dataset. The findings highlight the importance of age, blood pressure, and other features in predicting the disease. The findings can be used to support healthcare professionals in early detection and management of CKD, potentially leading to improved patient outcomes and quality of life.

