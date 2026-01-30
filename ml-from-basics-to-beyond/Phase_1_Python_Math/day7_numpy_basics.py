# Day 7 - NumPy Basics(arrays, operations, indexing)

'''
NumPy is a Python library used for working with arrays.
It also has functions for working in domain of linear algebra, fourier transform, and matrices.
NumPy was created in 2005 by Travis Oliphant. It is an open source project and you can use it freely.
NumPy stands for Numerical Python.
NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently.
This behavior is called locality of reference in computer science.

Why We Use NumPy:
1. NumPy arrays are faster and more compact than Python lists.
2. NumPy provides a large set of numeric datatypes to choose from.
3. NumPy arrays facilitate advanced mathematical and other types of operations on large numbers of data.
4. NumPy operations are implemented in C and Fortran. So they are faster than operations implemented in Python

Data Structures in NumPy:
1. ndarray: A multidimensional array object.

Basic Operations Covered:
1. Creating NumPy arrays
2. Array operations (addition, subtraction, multiplication, division, exponentiation, square root)
3. Joining and splitting arrays
4. Indexing and slicing
5. Searching, sorting, and filtering arrays

NumPy Documentation: https://numpy.org/doc/

'''

import numpy as np  # Importing the NumPy library

# Creating NumPy arrays
print("=== Creating NumPy Arrays ===")
array1 = np.array([1, 2, 3, 4, 5])
print("1D Array:", array1)
print(type(array1))

array2 = np.array([[1,2,3,4],[5,6,7,8]])
print("2D Array:", array2)

array3 = np.array([[1,]])
print("", array3)

# Array operations
print("\n=== Array Operations ===")
array_a = np.array([10, 20, 30, 40])
array_b = np.array([1, 2, 3, 4])
print("Array A:", array_a)
print("Array B:", array_b)
print("Addition:", array_a + array_b)
print("Subtraction:", array_a - array_b)
print("Multiplication:", array_a * array_b)
print("Division:", array_a / array_b)
print("Exponentiation:", array_a ** 2)
print("Square Root of Array A:", np.sqrt(array_a))
print("Square Root of Array B:", np.sqrt(array_b))

# Joining arrays
print("Join Arrays:", np.concatenate((array_a, array_b)))

# Splitting arrays - If the array has less elements than required, it will adjust from the end accordingly.
print("Split Array A into 2 parts:", np.array_split(array_a, 2))
print("Split Array B into 3 parts:", np.array_split(array_b, 3))

# Indexing and Slicing
print("\n=== Indexing and Slicing ===")
array_c = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("Original Array:\n", array_c)
print("Element at (1,2):", array_c[1, 2])  # Accessing element at row 1, column 2
print("First row:", array_c[0, :])  # Accessing the first row
print("Second column:", array_c[:, 1])  # Accessing the second column
print("Sliced Array (first two rows and first three columns):\n", array_c[:2, :3])

#Array Searching
print("\n=== Array Searching ===")
array_d = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("Original Array D:", array_d)
result = np.where(array_d > 50)
print("Indices of elements greater than 50:", result)

#Array Sorting
print("\n=== Array Sorting ===")
array_e = np.array([40, 10, 20, 50, 30, 60, 80, 70])
print("Original Array E:", array_e)
sorted_array = np.sort(array_e)
print("Sorted Array E:", sorted_array)

#Array Filtering
print("\n=== Array Filtering ===")
array_f = np.array([5, 15, 25, 35, 45, 55, 65, 75])
print("Original Array F:", array_f)
filtered_array = array_f[array_f > 50]
print("Filtered Array (elements greater than 50):", filtered_array)