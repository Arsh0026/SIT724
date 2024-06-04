#!/usr/bin/env python
# coding: utf-8

# In[301]:


import numpy as np
import pandas as pd
import editdistance
import itertools
import matplotlib.pyplot as plt
import radon


# In[302]:


def similarity_score(s1, s2):
    value = editdistance.eval(s1, s2)
    return 1 - value / max(len(s1), len(s2))


# In[303]:


def process_python_file(file_path):
    """
    Reads a Python file, processes its content as text.
    Args:
    file_path (str): The path to the Python file to read.

    Returns:
    dict: A dictionary with various metrics about the content.
    """
    with open(file_path, 'r') as file:
        content = file.readlines()  # Reads lines into a list

    # Example metrics - number of lines, number of non-empty lines, number of comments
    total_lines = len(content)
    non_empty_lines = sum(1 for line in content if line.strip())
    comments = sum(1 for line in content if line.strip().startswith('#'))

    metrics = {
        'total_lines': total_lines,
        'non_empty_lines': non_empty_lines,
        'comments': comments
    }

    return content


# In[135]:


file1 = 



# In[136]:


l1= similarity_score(process_python_file(main),process_python_file(file1))
l2= similarity_score(process_python_file(main),process_python_file(file2))
l3= similarity_score(process_python_file(main),process_python_file(file3))
l4= similarity_score(process_python_file(main),process_python_file(file4))
l5= similarity_score(process_python_file(main),process_python_file(file5))
l6= similarity_score(process_python_file(main),process_python_file(file6))
l7= similarity_score(process_python_file(main),process_python_file(file7))
l8= similarity_score(process_python_file(main),process_python_file(file8))
l9= similarity_score(process_python_file(main),process_python_file(file9))
l10= similarity_score(process_python_file(main),process_python_file(file10))
laven5 = [l1,l2,l3,l4,l5]
laven10 = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10]


# In[156]:


labels = ["Value"]

# Plotting the box plot with updated labels
plt.figure(figsize=(5, 4))
plt.boxplot(laven5, patch_artist=True, notch=False, labels=labels)
plt.title(' Lavenshtein Distance for Top 5 Generated Codes')
plt.ylabel('Similarity Score')
plt.grid(True)
plt.show()


# In[158]:


labels = ["Value"]

# Plotting the box plot with updated labels
plt.figure(figsize=(5, 4))
plt.boxplot(laven10, patch_artist=True, notch=False, labels=labels)
plt.title(' Lavenshtein Distance for Top 5 Generated Codes')
plt.ylabel('Similarity Score')
plt.grid(True)
plt.show()


# In[272]:


labels = ["Top 5 Generated Codes","Top 10 Generated Codes"]

# Plotting the box plot with updated labels
plt.figure(figsize=(5, 4))
plt.boxplot([laven5,laven10], patch_artist=True, notch=False, labels=labels)
plt.title(' Lavenshtein Distance for  Codes')
plt.ylabel('Similarity Score')
plt.grid(True)
plt.show()


# In[277]:


data_array=laven5
data_array = np.array(laven5)

# Calculate the mean
mean = np.mean(data_array)

# Calculate the median
median = np.median(data_array)

# Calculate the standard deviation
std_dev = np.std(data_array)

# Find the maximum value
max_val = np.max(data_array)

# Find the minimum value
min_val = np.min(data_array)

print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Maximum Value:", max_val)
print("Minimum Value:", min_val)


# In[278]:


data_array = np.array(laven10)

# Calculate the mean
mean = np.mean(data_array)

# Calculate the median
median = np.median(data_array)

# Calculate the standard deviation
std_dev = np.std(data_array)

# Find the maximum value
max_val = np.max(data_array)

# Find the minimum value
min_val = np.min(data_array)

print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Maximum Value:", max_val)
print("Minimum Value:", min_val)

