# Day 8 - Pandas Basics
'''
Pandas is a powerful and flexible open-source data analysis/manipulation library for Python.
It provides data structures and functions needed to manipulate structured data seamlessly.
Pandas is built on top of NumPy and is designed for practical data analysis in Python.
It was created by Wes McKinney in 2008.

Why We Use Pandas:
1. Easy handling of missing data.
2. Size mutability: columns can be inserted and deleted from DataFrame and higher dimensional objects.
3. Automatic and explicit data alignment.
4. Powerful, flexible group by functionality to perform split-apply-combine operations on data sets.
5. Time series-specific functionality: date range generation and frequency conversion.

The Primary Data Structures in Pandas are:
1. Series: A one-dimensional labeled array capable of holding any data type. Series can be thought of as a column in a table.
Series have homogeneous data types. If any element is of a different type, Pandas will upcast the entire Series to a common data type.
2. DataFrame: A two-dimensional labeled data structure with columns of potentially different types.
DataFrames can be thought of as a table or spreadsheet. DataFrames are mutable, meaning you can change their size and contents.
Values in a DataFrame can be of different data types (heterogeneous).

Basic Operations Covered:
1. Creating Series and DataFrames
2. Data selection and indexing
3. Handling missing data

Difference Between Series and DataFrame:
- Series is a one-dimensional array with labeled indices, while DataFrame is a two-dimensional table with rows and columns.
- Series can hold only one data type, whereas DataFrame can hold multiple data types across different columns.

Difference Between loc and iloc:
- loc is label-based indexing, which means you have to specify the names of the rows and columns that you want to select.
- iloc is integer position-based indexing, which means you have to specify the integer index positions of the rows and columns that you want to select.
Which is better depends on the use case. If you know the labels of the rows and columns, loc is more intuitive. If you are working with integer positions, iloc is more straightforward.

Where Series and DataFrames are Used:
- Series are often used for time series data, single columns of data, or when you need a simple one-dimensional array with labels.
- DataFrames are used for more complex data structures, such as datasets with multiple columns, relational data, or when performing data analysis tasks that require multiple dimensions.


Command to Install Pandas:
pip install pandas

Pandas Documentation: https://pandas.pydata.org/docs/
'''

import pandas as pd  # Importing the Pandas library

# Creating a Pandas Series
mySeries = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # Creating a Pandas Series
print("Pandas Series:", mySeries)
print("\nDisplaying the first few rows of the Series :\n",mySeries.head()) # Displaying the first few elements of the Series

#Indexing and Selecting Data from Series
print("Element at index 2 in Series:", mySeries[2])  # Accessing element at index 2

#Location-based Indexing - uses iloc
print("Element at index 3 using iloc:", mySeries.iloc[3])  # Accessing element at index 3 using iloc

# For multiple indices
print("Elements at indices 1, 3, and 4 using iloc:", mySeries.iloc[[1, 3, 4]])  # Accessing elements at multiple indices using iloc

#Conditional Selection
print("Elements greater than 45 in Series:\n", mySeries[mySeries > 45])  # Elements greater than 45
print("Elements greater than 45 and less than 75 in Series:\n", mySeries[(mySeries > 45) & (mySeries < 75)])  # Elements greater than 45 and less than 75

# Creating a Pandas DataFrame
data = {
    'Name': ['Kim', 'Bob', 'Choke', 'Dame'],
    'Age': [24, 29, 42, 32],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Berlin']
}
myDataFrame  = pd.DataFrame(data)

print("\nPandas DataFrame:\n", myDataFrame)
print("\nDisplaying the first few rows of the DataFrame:\n", myDataFrame.head())

#Indexing and Selecting Data from DataFrame
print("Names in DataFrame:\n", myDataFrame['Name'])  # Accessing the 'Name' column

#Location-based Indexing - uses iloc
print("Element at row 1, column 2 using iloc:", myDataFrame.iloc[1, 2])  # Accessing element at row 1, column 2 using iloc

# For multiple rows and columns
print("Elements at rows 0 and 2, columns 1 and 2 using iloc:\n", myDataFrame.iloc[[0, 2], [1, 2]])  # Accessing multiple rows and columns using iloc

#Conditional Selection
print("Rows where Age > 30:\n", myDataFrame[myDataFrame['Age'] > 30])  # Rows where Age is greater than 30

#loc-based Indexing - uses loc
print("Element at row 2, column 'City' using loc:", myDataFrame.loc[2, 'City'])  # Accessing element at row 2, column 'City' using loc

# For multiple rows and columns
print("Elements at rows 0 and 2, columns 'Name' and 'Age' using loc:\n", myDataFrame.loc[[0, 2], ['Name', 'Age']])  # Accessing multiple rows and columns using loc

#Handling Missing Data
data_with_nan = {
    'Name': ['Alice', 'Bob', None, 'David', 'Kevin', 'Ronnie', 'Amit'],
    'Age': [25, None, 30, 35, None, 27, 29],
    'Salary': [50000, 60000, 70000, 80000, 90000, 100000, 1500000],
    'City': ['New York', 'Los Angeles', 'Chicago', None, 'Berlin', None ,'India']
}
df_with_nan = pd.DataFrame(data_with_nan)
print("\nDataFrame with Missing Data:\n", df_with_nan)

# Handling missing data by filling with a specific value
df_filled = df_with_nan.fillna('Unknown')
print("\nDataFrame after filling missing data:\n", df_filled)

# Handling missing data by dropping rows with any missing values
df_dropped = df_with_nan.dropna()
print("\nDataFrame after dropping rows with missing data:\n", df_dropped)

# Handling missing data by dropping columns with any missing values
df_dropped_columns = df_with_nan.dropna(axis=1)
print("\nDataFrame after dropping columns with missing data:\n", df_dropped_columns)

# CSV Handling Example
# Saving DataFrame to CSV
df_with_nan.to_csv('data_with_nan.csv', index=False)
# Reading DataFrame from CSV
df_from_csv = pd.read_csv('data_with_nan.csv')
print("\nDataFrame read from CSV:\n", df_from_csv)

# Operations CSV Handling Example
# Calculating mean age, ignoring NaN values
mean_age = df_with_nan['Age'].mean()
print("\nMean Age (ignoring NaN values):", mean_age)

# Calculating total salary, ignoring NaN values
total_salary = df_with_nan['Salary'].sum()
print("Total Salary (ignoring NaN values):", total_salary)


# End of Day 8 - Pandas Basics