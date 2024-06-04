#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""""
1
1
Generate code to add a learning rate scheduler to the optimizer in the quantum hybrid neural network to improve training efficiency
"If available, write a function that will ensure the hybridNet model and all tensors are moved to the graphics card."
Data augmentation involves providing code to implement data augmentation in the DatasetHandler class.
"When data paths or labels can't be correctly loaded or parsed, create error handling in the DatasetHandler class."
"Suggest code modifications to allow dynamic adjustment of quantum circuit parameters, such as the number of qubits and layers, based on dataset features."
Generate code for visualization of training loss and validation accuracy using matplotlib or another plotting library.
"Improve the checkpoint saving mechanism to include more state information, such as the validation loss and learning rate, and generate code for automatic recovery from the last best state after a failure."
"After training, write code to perform more comprehensive model evaluations, including precision-recall curves and ROC curves, for each class."
Propose a method to integrate additional quantum layers into the hybridnet model and generate corresponding PyTorch compatible quantum circuit code.

"""


# In[ ]:


"""Importing necessary libraries and configurations."""
from config import *
from utils.DatasetHandler import DatasetHandler
from models.HybridNet import HybridNet
from qc.QiskitCircuit import QiskitCircuit
from utils.utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import os

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

#=======================================================================================================================
"""Check for GPU availability and set the device accordingly."""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'[*] Running on device: {device}')

#=======================================================================================================================
"""Initialize the Quantum Hybrid Neural Network."""
network = HybridNet().to(device)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

#=======================================================================================================================
"""Dynamic adjustment of quantum circuit parameters based on data features."""
def adjust_quantum_circuit(data_features):
    """Adjusts the number of qubits and layers dynamically."""
    num_qubits = max(2, min(10, int(np.log2(data_features) + 1)))  # Example logic for dynamic adjustment
    num_layers = max(1, min(5, data_features // 5))
    return QiskitCircuit(num_qubits, SIMULATOR, NUM_SHOTS, num_layers)

circuit = adjust_quantum_circuit(len(classes))
print(circuit.circuit.draw(output='text'))

#=======================================================================================================================
"""Enhance data handling with augmentation using the DatasetHandler class."""
class AugmentedDatasetHandler(DatasetHandler):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

handler_train = AugmentedDatasetHandler(training_root)
handler_val = DatasetHandler(validation_root)

#=======================================================================================================================
"""Training the model with real-time visualization of training metrics."""
fig, ax = plt.subplots()
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    network.train()
    total_loss = []
    for data, target in handler_train.get_loader():
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    train_loss = np.mean(total_loss)
    train_losses.append(train_loss)
    val_loss = validate_model(handler_val, network, device)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    # Update plots
    ax.clear()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.pause(0.1)

plt.show()

#=======================================================================================================================
"""Post-training evaluation with precision-recall and ROC curves."""
def model_evaluation(handler, model, device):
    """Evaluate the model and plot precision-recall and ROC curves."""
    model.eval()
    targets, outputs = [], []
    with torch.no_grad():
        for data, target in handler.get_loader():
            data, target = data.to(device), target.to(device)
            output = model(data)
            targets.extend(target.tolist())
            outputs.extend(output[:,1].tolist())  # Assuming binary classification for simplicity

    precision, recall, _ = precision_recall_curve(targets, outputs)
    fpr, tpr, _ = roc_curve(targets, outputs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(recall, precision)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.subplot(122)
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

#=======================================================================================================================
"""Integration of additional quantum layers into the HybridNet model."""
def integrate_quantum_layers(model, num_layers):
    """Integrates additional quantum layers into the HybridNet model."""
    model.quantum = QiskitCircuit(model.quantum.num_qubits, model.quantum.simulator, model.quantum.shots, num_layers)
    return model
