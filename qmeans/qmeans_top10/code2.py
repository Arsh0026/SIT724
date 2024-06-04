#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector
from sklearn.datasets import make_blobs  # To generate synthetic data

# Constants
k = 3  # Number of clusters
iterations = 50
threshold = 0.01  # Convergence threshold

# Generate synthetic data
data, _ = make_blobs(n_samples=100, centers=k, n_features=2, random_state=42)
data /= np.linalg.norm(data, axis=0)  # Normalize for quantum state preparation

# Initialize quantum registers
qreg = QuantumRegister(3)  # Two for the state vectors and one for ancillary qubit
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
    # Cluster assignment step
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
            # Calculate distances based on the probability of measuring |0>
            distances.append(1024 - result.get('1', 0))
        clusters[np.argmin(distances)].append(point)

    # Update centroids
    new_centroids = np.array([np.mean(clusters[i], axis=0) for i in range(k)])
    # Convergence check
    if np.linalg.norm(new_centroids - centroids) < threshold:
        print(f"Convergence reached after {it+1} iterations.")
        break
    centroids = new_centroids

# Display results
plt.figure(figsize=(8, 6))
for i, points in clusters.items():
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], s=50, label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, color='red', marker='*', label='Centroids')
plt.title('Quantum k-Means Clustering')
plt.legend()
plt.grid(True)
plt.show()

# Output final centroids and clustering evaluation
print("Final centroids:")
print(centroids)

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
import numpy as np

# Constants
k = 2  # Number of clusters
max_iterations = 50
convergence_threshold = 0.01
num_qubits = 3  # Assuming 3 qubits for simplicity

# Sample data points (classical data)
data_points = np.array([[0.1, 0.2], [0.4, 0.5], [0.8, 0.9]])

# Step 1: Initialize quantum circuit
def initialize_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits)
    # Initialize qubits in superposition or another desired initial state
    qc.h(range(num_qubits))
    return qc

# Step 2: Create deterministic cluster centroids (classical method, simple initialization)
def initialize_centroids(data_points, k):
    # Simple approach: choose first 'k' points as centroids
    return data_points[:k]

# Step 3: Convert data points into quantum states
def encode_data_to_quantum_state(data_point, num_qubits):
    qc = QuantumCircuit(num_qubits)
    # Example encoding: use real amplitudes for simplicity
    qc.initialize(data_point, range(num_qubits))
    return qc

# Step 4: Quantum routine to estimate distances (hypothetical function)
def quantum_distance_estimation(qc1, qc2):
    # Placeholder for quantum state overlap calculation
    backend = Aer.get_backend('statevector_simulator')
    job = execute([qc1, qc2], backend)
    result = job.result()
    sv1 = Statevector(result.get_statevector(qc1))
    sv2 = Statevector(result.get_statevector(qc2))
    return np.abs(sv1.inner(sv2))**2

# Main loop
def quantum_k_means(data_points, k, max_iterations, convergence_threshold):
    centroids = initialize_centroids(data_points, k)
    for iteration in range(max_iterations):
        clusters = {i: [] for i in range(k)}
        
        # Assign points to the nearest centroid
        for point in data_points:
            qc_point = encode_data_to_quantum_state(point, num_qubits)
            distances = []
            for centroid in centroids:
                qc_centroid = encode_data_to_quantum_state(centroid, num_qubits)
                distance = quantum_distance_estimation(qc_point, qc_centroid)
                distances.append(distance)
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(point)

        # Update centroids
        new_centroids = []
        for i in range(k):
            new_centroids.append(np.mean(clusters[i], axis=0))
        
        # Check convergence
        if np.linalg.norm(new_centroids - centroids) < convergence_threshold:
            break
        centroids = new_centroids

    return centroids

# Execute the algorithm
final_centroids = quantum_k_means(data_points, k, max_iterations, convergence_threshold)
print("Final centroids:")
print(final_centroids)


