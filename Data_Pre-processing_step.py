import pandas as pd
import numpy as np
import boto3
import statistics as st

# Read the ARFF file as plain text
file_path = "chronic_kidney_disease.arff"

with open(file_path, 'r') as f:
    arff_text = f.read()

# Extract the attribute section
col_start = arff_text.index("@relation") + len("@relation") + 1
col_end = arff_text.index("@data") + len("@data") + 1
column_str = arff_text[col_start:col_end]

# Get the attribue that will be used as column names in pandas dataframe
c = column_str.split("\n")
col = [i.split(" ")[1] for i in c if len(i.split(" ")) > 1]
col =[elem.replace("'", "") for elem in col]

# Split the data into rows and convert to a list of lists
data_start = arff_text.index("@data") + len("@data") + 1
data_str = arff_text[data_start:]
data = data_str.split("\n")
data = [i.split(",") for i in data]
data = data[1:401]

df = pd.DataFrame(data, columns= col)

##check for null values
print(df.isnull().sum())

#replace ? with null to get an overview of the null values
df.replace('?', np.nan,inplace=True)

#check the amount of null values
print(df.isnull().sum())

##replace numerical columns that have null values to zero
def numericalNan_to_zero(df, columns):
    df[columns] = df[columns].replace(np.nan, 0)
    return df

##convert numerical columns to floats
def convert_columns_to_floats(df, columns):
    df[columns] = df[columns].astype(float)
    return df

columns_to_convert = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

df = numericalNan_to_zero(df, columns_to_convert)
print(df[df['age']==0])

columns_to_convert = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

df = convert_columns_to_floats(df, columns_to_convert)

##write a function to replace the numerical columns with a value of 0 with the mean of the column
def replace_zero_with_mean(df):
    numerical_columns = df.select_dtypes(include=np.number).columns
    for column in numerical_columns:
        df[column] = df[column].replace(0, df[column].mean())

    return df
df = replace_zero_with_mean(df)

##write a function to replace the nan non numerical columns with the mode of the column
def replace_nan_with_mode(df):
    non_numerical_columns = df.select_dtypes(exclude=np.number).columns
    for column in non_numerical_columns:
        df[column] = df[column].replace(np.nan, st.mode(df[column]))

    return df
df = replace_nan_with_mode(df)

print(df.isnull().sum())

##save to csv
df.to_csv('chronic_kidney_disease.csv', index=False)

#upload to an aws bucket
bucket = "github-data-bucket"
file_path = "/home/krissemmy/hamoye/chronic_kidney_disease.csv"
file_name = "chronic_kidney_disease.csv"

s3 = boto3.client('s3')
s3.upload_file(file_path, bucket, file_name)
print("File successfully uploaded to AWS s3")
