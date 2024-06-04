#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Initialize classical and quantum register.
cluster centroids can be set randomly or using a deterministic method.
Classical data points should be translated into quantum states.
The maximum number and convergence threshold should be set.
Continue until you reach the maximum iteration.
The nearest cluster centroid will be assigned each data point's location.
The cluster centroids have been updated.
Check to see if there is convergence.
The final cluster centroids are in the quantum register.
Classical representations can be obtained by coding quantum cluster centroids.
Using metrics such as within-cluster sum of squares, evaluate clustering quality.
The evaluation results and final cluster centroids should be output.

"""


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from sklearn.datasets import make_blobs  # To generate synthetic data
from sklearn.metrics import silhouette_score

# Constants
k = 3  # Number of clusters
iterations = 50
threshold = 0.01  # Convergence threshold

# Generate synthetic data
data, true_labels = make_blobs(n_samples=100, centers=k, n_features=2, random_state=42)
data /= np.linalg.norm(data, axis=0)  # Normalize for quantum state preparation

# Initialize quantum registers
qreg = QuantumRegister(3)  # Two for the state vectors and one for ancilla
creg = ClassicalRegister(1)
qc = QuantumCircuit(qreg, creg)

# Function to encode points into quantum states
def encode_point(point, qc, qubits):
    qc.initialize(point.tolist(), qubits)

# Function to calculate the swap test
def swap_test(qc, reg1, reg2, ancilla):
    qc.h(ancilla)
    qc.cswap(ancilla, reg1, reg2)
    qc.h(ancilla)
    qc.measure(ancilla, 0)
    return qc

# Initialize centroids
centroids = np.random.randn(k, data.shape[1])  # Random initial centroids

# Quantum k-Means algorithm
for it in range(iterations):
    clusters = {i: [] for i in range(k)}
    for point in data:
        distances = []
        for centroid in centroids:
            qc = QuantumCircuit(qreg, creg)
            encode_point(point, qc, [0])
            encode_point(centroid, qc, [1])
            qc = swap_test(qc, 0, 1, 2)
            backend = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend, shots=1024)
            result = job.result().get_counts()
            distances.append(1024 - result.get('1', 0))
        clusters[np.argmin(distances)].append(point)

    new_centroids = np.array([np.mean(clusters[i], axis=0) for i in range(k)])
    if np.linalg.norm(new_centroids - centroids) < threshold:
        print(f"Convergence reached after {it+1} iterations.")
        break
    centroids = new_centroids

# Evaluate clustering quality using Within-Cluster Sum of Squares (WCSS)
wcss = sum(np.linalg.norm(np.array(clusters[i]) - centroids[i])**2 for i in range(k))

# Output final centroids and clustering evaluation
print("Final centroids:")
print(centroids)
print(f"Within-cluster sum of squares (WCSS): {wcss}")
print(f"Silhouette Score: {silhouette_score(data, [item for sublist in [ [i]*len(clusters[i]) for i in range(k)] for item in sublist])}")

# Plot the data and centroids
plt.figure(figsize=(8, 6))
for i, points in clusters.items():
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], s=50, label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, color='red', marker='*', label='Centroids')
plt.title('Quantum k-Means Clustering')
plt.legend()
plt.grid(True)
plt.show()


#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import pi
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute

# Get the data from the .csv file
df = pd.read_csv('data.csv')

data = df.values[:, 1:]



# In[19]:


df.head()


# In[20]:


# Number of clusters
k = 3
# Number of training data
n = data.shape[0]
# Number of features in the data
c = data.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = np.random.randn(k,c)*std + mean

# Static data to test
centers = np.array([[-0.25, 0.2], [0, -0.1], [0.25, 0.35]])
print(centers)

# Plot the data and the centers generated as random
colors=['green', 'blue', 'black']
for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7)
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)


# In[21]:


def point_centroid_distances(point, centroids):
    
    # Calculating theta and phi values
    phi_list = [((x + 1) * pi / 2) for x in [point[0], centroids[0][0], centroids[1][0], centroids[2][0]]]
    theta_list = [((x + 1) * pi / 2) for x in [point[1], centroids[0][1], centroids[1][1], centroids[2][1]]]

    # Create a 2 qubit QuantumRegister - two for the vectors, and 
    # one for the ancillary qubit
    qreg = QuantumRegister(3, 'qreg')

    # Create a one bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(1, 'creg')

    qc = QuantumCircuit(qreg, creg, name='qc')

    # Get backend using the Aer provider
    backend = Aer.get_backend('qasm_simulator')

    # Create list to hold the results
    results_list = []

    # Estimating distances from the new point to the centroids
    for i in range(1, 4):
        # Apply a Hadamard to the ancillary
        qc.h(qreg[2])

        # Encode new point and centroid
        qc.u3(theta_list[0], phi_list[0], 0, qreg[0])           
        qc.u3(theta_list[i], phi_list[i], 0, qreg[1]) 

        # Perform controlled swap
        qc.cswap(qreg[2], qreg[0], qreg[1])
        # Apply second Hadamard to ancillary
        qc.h(qreg[2])

        # Measure ancillary
        qc.measure(qreg[2], creg[0])

        # Reset qubits
        qc.reset(qreg)

        # Register and execute job
        job = execute(qc, backend=backend, shots=5000)
        result = job.result().get_counts(qc)
        try:
            results_list.append(result['1'])
        except:
            results_list.append(0)


    return results_list


# In[22]:


centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)
upper_error = error + 1

# When, after an update, the estimate of that center stays the same, exit loop
while (error + 0.02) < upper_error:
    # Measure the distance to every center
    
    distances = np.array(list(map(lambda x: point_centroid_distances(x, centers), data)))

    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    upper_error = deepcopy(error)
    error = np.linalg.norm(centers_new - centers_old)
    if error < 0.02:
        break
centers_new


# In[ ]:


# Plot the data and the centers generated as random
for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)


# In[ ]:


# Find the Euclidean distance between Quantum States    
def state_distance(state1, state2):
    # Create a 2 qubit QuantumRegister - two for the vectors, and 
    # one for the ancillary qubit
    qreg = QuantumRegister(3, 'qreg')

    # Create a one bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(1, 'creg')

    qc = QuantumCircuit(qreg, creg, name='qc')

    # Get backend using the Aer provider
    backend = Aer.get_backend('qasm_simulator')

    # Create list to hold the results
    results_list = []

    # Apply a Hadamard to the ancillary
    qc.h(qreg[2])

    # Encode new point and centroid
    qc.initialize(state1, qreg[0])           
    qc.initialize(state2, qreg[1]) 

    # Perform controlled swap
    qc.cswap(qreg[2], qreg[0], qreg[1])
    # Apply second Hadamard to ancillary
    qc.h(qreg[2])

    # Measure ancillary
    qc.measure(qreg[2], creg[0])

    # Reset qubits
    qc.reset(qreg)

    # Register and execute job
    job = execute(qc, backend=backend, shots=5000)
    result = job.result().get_counts(qc)
    try:
        results_list.append(result['1'])
    except:
        results_list.append(0)

    return results_list




