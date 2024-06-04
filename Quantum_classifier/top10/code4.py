#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
3
3
Generate code to add a learning rate scheduler to the optimizer in the quantum hybrid Neural Network to improve training efficiency.
"If available, write a function that will ensure the hybridNet model and all tensors are moved to the GPUs."
Code to implement data augmentation in the DatasetHandler class for the training dataset to enhance model generalizability.
"If data paths or labels can't be correctly loaded or parsed, create error handling in the DatasetHandler class."
"Suggest code modifications to allow dynamic adjustment of quantum circuit parameters, such as the number of qubits and layers based on dataset features."
Generate code for visualization of training loss and validation accuracy using matplotlib or another plot library.
"Improve the checkpoint saving mechanism to include more state information, such as the validation loss and the learning rate, and generate code for automatic recovery from the last best state after a failure"
"Code should be written to perform more comprehensive model evaluations after training, including precision-recall curves and ROC curves."
"Propose a method to integrate additional quantum layers into the hybridnet model, and generate corresponding PyTorch compatible quantum circuit code."

"""


# In[ ]:


""" Quantum Hybrid Neural Network Setup with Advanced Features """
import torch
import torch.nn as nn
import torch.optim as optim
from models.HybridNet import HybridNet
from qc.QiskitCircuit import QiskitCircuit
from utils.DatasetHandler import DatasetHandler
from utils.utils import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Set up for suppressing warnings and logging
import warnings
warnings.filterwarnings('ignore')

# Configuration imports
from config import *

# Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def to_device(data, device):
    """ Recursively move tensors to the device specified. """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Neural Network and Optimizer Configuration
network = HybridNet().to(device)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Data Handling with Augmentation and Error Handling
class RobustDatasetHandler(DatasetHandler):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform=transform)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]) if transform is None else transform
    
    def load_data(self):
        try:
            return super().load_data()
        except IOError as e:
            print(f"Failed to load data: {e}")
            return None

# Dynamic Quantum Circuit Configuration
def configure_quantum_circuit(features):
    """ Dynamically configure the quantum circuit based on input features. """
    num_qubits = max(2, int(features / 2))
    return QiskitCircuit(num_qubits, SIMULATOR, NUM_SHOTS)

# Training with Visualization
def train_network(epochs, network, train_loader, val_loader, optimizer, scheduler):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        network.train()
        total_loss = 0
        for data, targets in train_loader:
            data, targets = to_device(data, device), to_device(targets, device)
            optimizer.zero_grad()
            outputs = network(data)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        network.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for data, targets in val_loader:
                data, targets = to_device(data, device), to_device(targets, device)
                outputs = network(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_accuracy = 100 * correct / total
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(val_accuracy)
        scheduler.step(total_loss / len(train_loader))

        # Update plots
        ax1.plot(train_losses, label='Train Loss' if epoch == 0 else "")
        ax2.plot(val_accuracies, label='Validation Accuracy' if epoch == 0 else "")
        if epoch == 0:
            ax1.legend()
            ax2.legend()
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()

# Enhanced Checkpoint Saving
def save_checkpoint(state, filename='model_checkpoint.pth'):
    print(f"Saving checkpoint: {filename}")
    torch.save(state, filename)

# Comprehensive Model Evaluation
def evaluate_model(model, test_loader):
    """ Evaluate the model with precision-recall and ROC curves. """
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = to_device(data, device), to_device(targets, device)
            outputs = model(data)
            probabilities = nn.functional.softmax(outputs, dim=1)[:, 1]
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt

