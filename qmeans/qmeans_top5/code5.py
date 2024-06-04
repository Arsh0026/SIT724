#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Contribute quantum and classical register.
Cluster centroids can be used using a deterministic method or randomly.
Represent classical data points in quantum states.
The maximum number of iterations and convergence threshold is set.
Continue until convergence or the maximum iteration.
Each data point should be assigned to the nearest cluster.
Cluster centroids are being updated.
You should check for convergence.
The final cluster centroids can be found in the quantum register.
Classical representations are obtained by coding quantum cluster centroids.
Evaluate clustering quality with metrics such as the within-cluster sum of squares.
The final clusters and evaluation results should be output.
"""


# In[ ]:


import numpy as np
from sklearn.datasets import make_blobs
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector

# Function to normalize data
def normalize_data(data):
    return data / np.linalg.norm(data, axis=0)

# Generate synthetic data
data, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
data = normalize_data(data)

# Constants for Quantum k-Means
k = 3  # Number of clusters
max_iterations = 100
convergence_threshold = 1e-3

# Quantum circuit setup
qreg = QuantumRegister(3)  # Additional qubits for quantum states
creg = ClassicalRegister(1)
circuit = QuantumCircuit(qreg, creg)

# Initialize centroids randomly or deterministically
centroids = normalize_data(np.random.randn(k, data.shape[1]))

def initialize_state(vector, qubits):
    """ Initialize the quantum state corresponding to the vector. """
    norm = np.linalg.norm(vector)
    if norm > 0:
        return Statevector(vector / norm).data
    return Statevector([1, 0] * (len(qubits) // 2)).data

def apply_swap_test(circuit, data_qubit, centroid_qubit, ancilla_qubit):
    """ Apply the swap test to estimate the similarity (distance) between two quantum states. """
    circuit.h(ancilla_qubit)
    circuit.cswap(ancilla_qubit, data_qubit, centroid_qubit)
    circuit.h(ancilla_qubit)
    circuit.measure(ancilla_qubit, 0)
    return circuit

# Quantum k-Means main loop
for iteration in range(max_iterations):
    cluster_assignments = []
    for point in data:
        distances = []
        for centroid in centroids:
            qc = QuantumCircuit(qreg, creg)
            # Initialize quantum states for the point and centroid
            point_state = initialize_state(point, [1])
            centroid_state = initialize_state(centroid, [2])
            qc.initialize(point_state, [1])
            qc.initialize(centroid_state, [2])
            # Apply the swap test
            apply_swap_test(qc, 1, 2, 0)
            # Execute the circuit
            backend = Aer.get_backend('qasm_simulator')
            result = execute(qc, backend, shots=1024).result()
            counts = result.get_counts(qc)
            similarity = counts.get('0', 0) / 1024
            distances.append(1 - similarity)
        cluster_assignments.append(np.argmin(distances))

    # Update centroids
    new_centroids = []
    for i in range(k):
        points = data[np.array(cluster_assignments) == i]
        if len(points) > 0:
            new_centroids.append(np.mean(points, axis=0))
        else:
            new_centroids.append(centroids[i])  # Avoid empty cluster
    new_centroids = normalize_data(np.array(new_centroids))

    # Check for convergence
    if np.linalg.norm(new_centroids - centroids) < convergence_threshold:
        print(f"Converged after {iteration+1} iterations.")
        centroids = new_centroids
        break
    centroids = new_centroids

# Output final results
print("Final centroids:")
print(centroids)

# Evaluate clustering quality (Within-cluster sum of squares)
wcss = sum(np.linalg.norm(data[np.array(cluster_assignments) == i] - centroid)**2 
           for i, centroid in enumerate(centroids))
print(f"Within-cluster sum of squares: {wcss}")

