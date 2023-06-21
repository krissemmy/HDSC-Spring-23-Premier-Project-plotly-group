#!/usr/bin/env python
# coding: utf-8

# # 1. Import the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler



# # 2. Load the dataset

# In[2]:


df = pd.read_csv('https://github-data-bucket.s3.amazonaws.com/chronic_kidney_disease.csv')
df.head()


# # 3. EDA

# In[3]:


df.info()


# In[4]:


# checking the cloumns in the data
df.columns


# In[5]:


df.nunique()


# Each row in the dataset represents a patient, and the columns correspond to different features or attributes of the patients. Here's a breakdown of the columns you've mentioned:
# 
# - `age`: The age of the patient (in years)
# - `bp`: Blood pressure of the patient (measured in mmHg)
# - `sg`: Specific gravity of urine
# - `al`: Albumin present in the urine
# - `su`: Sugar present in the urine
# - `rbc`: Red blood cells in urine (either "normal" or "abnormal")
# - `pc`: Pus cell in urine (either "normal" or "abnormal")
# - `pcc`: Pus cell clumps in urine (either "present" or "not present")
# - `ba`: Bacteria present in urine (either "present" or "not present")
# - `bgr`: Blood glucose random (measured in mg/dL)
# - `pcv`: Packed cell volume
# - `wbcc`: White blood cell count (measured in cells/cubic mm)
# - `rbcc`: Red blood cell count (measured in millions/cubic mm)
# - `htn`: Whether the patient has hypertension (either "yes" or "no")
# - `dm`: Whether the patient has diabetes mellitus (either "yes" or "no")
# - `cad`: Whether the patient has coronary artery disease (either "yes" or "no")
# - `appet`: Patient's appetite (either "good" or "poor")
# - `pe`: Presence of pedal edema (either "yes" or "no")
# - `ane`: Presence of anemia (either "yes" or "no")
# - `class`: The target variable, indicating the presence or absence of chronic kidney disease (either "ckd" or "notckd")
# 
# 
# 
# 

# In[6]:


# checking the unique values in the target values
df['class'].unique()


# **"ckd"** refers to **chronic kidney disease**, while **"notckd"** indicates the absence of **chronic kidney disease**. The "class" column serves as the target variable that classifies whether a patient has chronic kidney disease or not based on the given features.

# In[7]:


# shape of the data
df.shape


# In[8]:


# check null values
df.isnull().sum()


# In[9]:


# check for the Target Variable Distribution
target_counts = df['class'].value_counts()
print(target_counts)


plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='class')
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[10]:


# check for the outliers
plt.figure(figsize=(15, 6))
sns.boxplot(data=df)
plt.title('Box Plot of Numerical Variables')
plt.xticks(rotation=90)
plt.show()


# In[11]:


df.describe()


# In[12]:


# check the correlation of the data
correlation_matrix = df.corr()
plt.figure(figsize=(20, 8))
sns.heatmap(correlation_matrix, annot=True,fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[13]:


# Distribution of age
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', bins=20)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[14]:


# Scatter plot of blood pressure (bp) vs. glucose random measurement (bgr)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='bp', y='bgr', hue='class')
plt.title('Blood Pressure vs. Glucose Random Measurement')
plt.xlabel('Blood Pressure')
plt.ylabel('Glucose Random Measurement')
plt.legend(title='Class')
plt.show()


# In[15]:


# Box plot of serum creatinine (sc) by presence of hypertension (htn)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='htn', y='sc')
plt.title('Serum Creatinine by Hypertension')
plt.xlabel('Hypertension')
plt.ylabel('Serum Creatinine')
plt.show()


# In[16]:


#  visualize pairwise relationships between multiple variables, highlighting different classes.
plt.figure(figsize=(10, 8))
sns.pairplot(data=df, vars=['age', 'bgr', 'sc', 'hemo'], hue='class')
plt.title('Pairwise Relationships')
plt.show()


# In[17]:


# visualize the distribution of a continuous variable (age) across different classes
plt.figure(figsize=(10, 8))
sns.violinplot(data=df, x='class', y='age')
plt.title('Age Distribution by Class')
plt.xlabel('Class')
plt.ylabel('Age')
plt.show()


# In[18]:


# plot displays the average hemoglobin level for each class, along with error bars representing the standard deviation.
plt.figure(figsize=(10, 8))
sns.barplot(data=df, x='class', y='hemo', ci='sd')
plt.title('Average Hemoglobin Level by Class')
plt.xlabel('Class')
plt.ylabel('Hemoglobin Level')
plt.show()


# # 4. Data Cleaning and Preprocessing:

# In[19]:


df.info()


# In[20]:


# Select the categorical variables to encode
categorical_vars = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical variables
for var in categorical_vars:
    df[var] = label_encoder.fit_transform(df[var])


# In[20]:





# In[21]:


# Feature Engineering


# Calculate Mean Corpuscular Volume (MCV)
pcv_values = np.nan_to_num(df['pcv'].values)
rbc_values = np.nan_to_num(df['rbc'].values)
mcv_values = np.divide(pcv_values, rbc_values)
mcv_values = np.where(np.isfinite(mcv_values), mcv_values, 0)  # Replace infinity with 0
df['mcv'] = np.round(mcv_values, 2)

# Calculate Glucose-to-Blood Pressure Ratio
df['glucose_bp_ratio'] = (df['bgr'] / df['bp']).apply(lambda x: round(x, 2) if np.isfinite(x) else x)

# Total Blood Cell Count
df['total_blood_cell_count'] = df['rbcc'] + df['wbcc']

# Anemia Indicator
df['anemia'] = np.where(df['hemo'] < 12, 1, 0)

# Blood Pressure Category
#df['bp_category'] = pd.cut(df['bp'], bins=[0, 120, 130, np.inf], labels=['0', '1', '2'], right=False)
df['bp_category'] = pd.cut(df['bp'], bins=[0, 120, 130, np.inf], labels=[0, 1, 2], right=False).astype(int)

# Display the updated dataset with new features
df.head()




# - **Mean Corpuscular Volume (MCV)**: the patient's red blood cell count (rbc) and packed cell volume (pcv), you can calculate the mean corpuscular volume (MCV), which represents the average volume of red blood cells. MCV can be calculated using the formula: MCV = pcv / rbc.
# 
# - **Glucose-to-Blood Pressure Ratio:** the ratio between the blood glucose random level (bgr) and the blood pressure (bp) of the patients. This ratio can provide insights into the relationship between glucose levels and blood pressure.
# 
# - **Total Blood Cell Count:** Combination of the red blood cell count (rbcc) and white blood cell count (wbcc) to create a new feature representing the total blood cell count.
# 
# - **Anemia Indicator**: a binary feature indicating the presence or absence of anemia based on the hemo (hemoglobin) level.
# 
# - **Blood Pressure Category:** Categorize blood pressure (bp) values into different categories such as **"low = 0," "normal = 1," and "high = 3."**

# In[22]:


df.columns


# In[23]:


df.info()


# In[24]:


# Separate the features (X) and target variable (y)
X = df.drop('class', axis=1)
y = df['class']


# In[25]:


# Define the RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Apply undersampling to the feature matrix X and target variable y
X_res, y_res = rus.fit_resample(X, y)


# In[26]:


# Convert the target variable to a DataFrame (assuming it is a pandas Series)
y_res_df = pd.DataFrame(y_res, columns=['class'])

# Count the occurrences of each class in the target variable
class_counts = y_res_df['class'].value_counts()

# Print the class counts
print(class_counts)


# Create a bar chart to visualize the class balance
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution after SMOTE Oversampling')
plt.show()



# In[27]:


# Split the data into train and test sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print("Train set shape:", X_train_s.shape, y_train_s.shape)
print("Test set shape:", X_test_s.shape, y_test_s.shape)


# In[27]:





# # 5. Choose, train and avaluation of Model

# In[28]:


# Create an instance of Logistic Regression
logreg = LogisticRegression()

# Fit the model on the training data
logreg.fit(X_train_s, y_train_s)

# Make predictions on the test data
test_preds = logreg.predict(X_test_s)

# Print the classification report for the test set
test_accuracy = accuracy_score(y_test_s, test_preds)
print("Test Accuracy:", test_accuracy)
print("Classification Report for Test Set:")
print(classification_report(y_test_s, test_preds))

# Make predictions on the training data
train_preds = logreg.predict(X_train_s)

# Print the classification report for the training set
train_accuracy = accuracy_score(y_train_s, train_preds)
print("Training Accuracy:", train_accuracy)
print("Classification Report for Training Set:")
print(classification_report(y_train_s, train_preds))


# 

# 

# In[28]:





# In[28]:





# In[29]:


# Create the Random Forest classifier

clf = RandomForestClassifier(n_estimators=100)

# Train the classifier
clf.fit(X_train_s, y_train_s)

# Predict on the training set
train_preds4 = clf.predict(X_train_s)
train_accuracy4 = accuracy_score(y_train_s, train_preds4)
print("Training Accuracy:", train_accuracy4)

# Print the classification report for the training set
print("Classification Report for Training Set:")
print(classification_report(y_train_s, train_preds4))


# Predict on the test set
test_preds4 = clf.predict(X_test_s)
test_accuracy4 = accuracy_score(y_test_s, test_preds4)
print("Test Accuracy:", test_accuracy4)

# Classification report for test set
print("Classification Report for Test Set:")
print(classification_report(y_test_s, test_preds4))


# In[29]:





# In[30]:


# Create the gradient boosting classifier

gbc = GradientBoostingClassifier()

# Train the classifier
gbc.fit(X_train_s, y_train_s)

# Predict on the training set
train_preds1 = gbc.predict(X_train_s)
train_accuracy1 = accuracy_score(y_train_s, train_preds1)
print("Training Accuracy:", train_accuracy1)

# Print the classification report for the training set
print("Classification Report for Training Set:")
print(classification_report(y_train_s, train_preds1))

# Predict on the test set
test_preds1 = gbc.predict(X_test_s)
test_accuracy1 = accuracy_score(y_test_s, test_preds1)
print("Test Accuracy:", test_accuracy1)

# Classification report for test set
print("Classification Report for Test Set:")
print(classification_report(y_test_s, test_preds1))


# In[30]:





# In[31]:


# Create a decision tree classifier

dtree = DecisionTreeClassifier()

# Train the classifier
dtree.fit(X_train_s, y_train_s)

# Predict on the training set
train_preds2 = dtree.predict(X_train_s)
train_accuracy2 = accuracy_score(y_train_s, train_preds2)
print("Training Accuracy:", train_accuracy2)

# Print the classification report for the training set
print("Classification Report for Training Set:")
print(classification_report(y_train_s, train_preds2))

# Predict on the test set
test_preds2 = dtree.predict(X_test_s)
test_accuracy2 = accuracy_score(y_test_s, test_preds2)
print("Test Accuracy:", test_accuracy2)

# Classification report for test set
print("Classification Report for Test Set:")
print(classification_report(y_test_s, test_preds2))


# In[31]:





# In[32]:


# Create a decision tree classifier


svc = SVC()

# Train the classifier
svc.fit(X_train_s, y_train_s)

# Predict on the training set
train_preds3 = svc.predict(X_train_s)
train_accuracy3 = accuracy_score(y_train_s, train_preds3)
print("Training Accuracy:", train_accuracy3)

# Print the classification report for the training set
print("Classification Report for Training Set:")
print(classification_report(y_train_s, train_preds3))

# Predict on the test set
test_preds3 = dtree.predict(X_test_s)
test_accuracy3 = accuracy_score(y_test_s, test_preds3)
print("Test Accuracy:", test_accuracy3)

# Classification report for test set
print("Classification Report for Test Set:")
print(classification_report(y_test_s, test_preds3))


# In[32]:





# In[33]:


# Create a dictionary to store the accuracies for each model

accuracies = {
    'Logistic Regression': test_accuracy,
    'Random Forest classifier': test_accuracy4,
    'Gradient Boosting': test_accuracy1,
    'Decision Tree': test_accuracy2,
    'SVC': test_accuracy3
}

# Find the model with the highest accuracy
best_model = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model]

# Print the best model and its accuracy
print("Best Model:", best_model)
print("Best Accuracy:", best_accuracy)


# In[35]:


# serialize and save the trained model to a file

import pickle

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file)


# In[1]:


pip install nbconvert


# In[2]:


jupyter nbconvert --to script Plot-Hamoye.ipynb

