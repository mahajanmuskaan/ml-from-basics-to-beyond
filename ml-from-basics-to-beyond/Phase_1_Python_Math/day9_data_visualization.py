# Day 9 - Data Visualization with Matplotlib and Seaborn - Basics

'''
Matplotlib and Seaborn are powerful libraries in Python for data visualization. They allow you to create a wide variety of static, animated, and interactive plots to help understand and communicate data insights effectively.
With Matplotlib, you can create basic plots like line graphs, bar charts, histograms, and scatter plots. 
Seaborn builds on top of Matplotlib and provides a high-level interface for drawing attractive and informative statistical graphics.

Why Data Visualization?
Data visualization is crucial in data analysis as it helps to:
1. Understand Data: Visualizations can reveal patterns, trends, and outliers that may not be apparent in raw data.
2. Communicate Findings: Visuals can effectively convey complex information in a more digestible format.
3. Explore Relationships: Visualizations can help identify relationships between variables, aiding in hypothesis generation and testing.

Who Created Matplotlib and Seaborn?
Matplotlib was created by John D. Hunter in 2003. Seaborn was developed by Michael Waskom and is built on top of Matplotlib, providing a more user-friendly interface for creating statistical graphics.

What core language or component is it built on?
Both Matplotlib and Seaborn are built on Python, leveraging its capabilities for data manipulation and analysis. They also utilize NumPy for numerical operations and Pandas for data handling.

Important Functions and Methods of Matplotlib and Seaborn:
1. plt.plot(): Creates a line plot.
2. plt.bar(): Creates a bar chart.
3. plt.hist(): Creates a histogram.
4. plt.scatter(): Creates a scatter plot.
5. sns.heatmap(): Creates a heatmap for visualizing matrix data.
6. sns.pairplot(): Creates a matrix of scatter plots for pairwise relationships in a dataset.
7. plt.xlabel(), plt.ylabel(), plt.title(): Add labels and title to plots.
8. plt.show(): Displays the plot.

 When to use which plot?
1. Use Line Plots for continuous data to show trends over time.
2. Use Bar Charts for categorical data to compare different groups.
3. Use Histograms to visualize the distribution of a single numerical variable.
4. Use Scatter Plots to explore relationships between two numerical variables.
5. Use Heatmaps to visualize correlations or intensity in matrix data.
6. Use Pairplots to examine relationships between multiple variables in a dataset.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

