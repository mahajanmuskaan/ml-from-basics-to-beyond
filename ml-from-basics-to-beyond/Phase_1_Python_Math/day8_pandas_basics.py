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
