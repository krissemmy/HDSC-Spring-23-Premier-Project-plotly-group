# HDSC-Spring-23-Premier-Project-plotly-group

#The data is available at an [AWS s3 bucket](https://github-data-bucket.s3.amazonaws.com/chronic_kidney_disease.csv)


# Chronic Kidney Disease Prediction

## Introduction
This project aims to predict the chances of being diagnosed with chronic kidney disease using a provided dataset. The goal is to address this highly fatal health problem and contribute to its resolution.

## Library Importation
In this section, we import the necessary libraries and packages required for data analysis, modeling, and evaluation. The common libraries used in this project include:

- Pandas for data manipulation and analysis
- NumPy for numerical operations
- Matplotlib and Seaborn for data visualization
- Scikit-learn for machine learning algorithms and evaluation metrics

## Loading the Dataset
The dataset for this project is obtained through a link provided by the data engineering team. We load the dataset into a Pandas DataFrame using appropriate functions or methods.

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

- Choosing a logistic regression classification model based on the given data
- Fitting the model to the training data
- Making predictions on the test data
- Evaluating the model's performance on the training and testing datasets using appropriate evaluation metrics
- Iterating this process on four other classification models (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, SVC)
- Choosing the model with the best accuracy

## Saving the Model for Deployment
After training and evaluating the model, we save it in a serialized format for deployment. The serialized model can be easily loaded and used for making predictions in a production environment. The steps involved in saving the model are:

- Serializing the trained model using Python's pickle library
- Saving the serialized model to a specified file or storage location

Please note that the above README provides a general outline and explanation of the code.
