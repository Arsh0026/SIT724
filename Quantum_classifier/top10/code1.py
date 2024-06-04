


"""
Generate code to add a learning rate scheduler to the optimizer in the quantum hybrid neural network to improve training efficiency.
"If available, write a function that ensures the hybridNet model and all tensors are moved to the graphics card."
Data augmentation can be done by providing code to implement data augmentation in the DatasetHandler class.
There are scenarios where data paths or labels cannot be correctly loaded or parsed.
Suggest code modifications to allow dynamic adjustment of quantum circuit parameters such as the number of qubits and layers based on dataset features.
Generate code for real-time visualization of training loss and validation accuracy using matplotlib or another library.
"Improve the checkpoint saving mechanism to include more state information, such as the validation loss and the learning rate, and generate code for automatic recovery from the last best state after a failure."
"After training, write code to perform more comprehensive model evaluations, including precision-recall curves and ROC curves for each class."
Propose a method to integrate additional quantum layers into the hybridNet model and generate corresponding PyTorch compatible quantum circuit code.
"""





from config import *
from utils.DatasetHandler import DatasetHandler
from models.HybridNet import HybridNet
from qc.QiskitCircuit import QiskitCircuit
from utils.utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import torch.optim as optim
import torch.nn as nn
import torch
import os
import torchvision.transforms as transforms

# Suppressing warning
import warnings
warnings.filterwarnings('ignore')

#=======================================================================================================================
print('\n[%] Checking for the availability of GPUs')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\t [*] Running on device: {}'.format(device))

#=======================================================================================================================
print('\n[%] Initialize Quantum Hybrid Neural Network')
network = HybridNet().to(device)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#=======================================================================================================================
print('\n[%] Adjusting Quantum Circuit based on Data')
# Dynamic quantum circuit adjustment function
def adjust_quantum_circuit(data_features):
    num_qubits = int(np.ceil(np.log2(data_features)))
    return QiskitCircuit(num_qubits, SIMULATOR, NUM_SHOTS)

circuit = adjust_quantum_circuit(len(classes))  # Example adjustment based on number of classes
print(circuit.circuit.draw(output='text'))

#=======================================================================================================================
print('\n[%] Enhancing Data Handling with Augmentation')
# Update to DatasetHandler to include data augmentation
class AugmentedDatasetHandler(DatasetHandler):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])

handler_train = AugmentedDatasetHandler(training_root)
handler_val = DatasetHandler(validation_root)

#=======================================================================================================================
print('\n[%] Starting Training with Real-time Visualization')
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
train_losses, val_losses = [], []

if LOAD_CHECKPOINT:
    checkpoint = torch.load(MODEL_SAVE_PATH)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print('\t[*] Checkpoint loaded')
else:
    start_epoch = 0

for epoch in range(start_epoch, EPOCHS):
    network.train()
    total_loss = []
    for data, target in handler_train:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    scheduler.step()
    train_losses.append(np.mean(total_loss))
    val_loss = validate_model(handler_val, network, device)
    val_losses.append(val_loss)

    # Real-time plotting
    ax[0].plot(train_losses, label='Train Loss')
    ax[1].plot(val_losses, label='Validation Loss')
    plt.pause(0.05)

plt.show()

#=======================================================================================================================
def validate_model(data_handler, model, device):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for data, target in data_handler:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            val_losses.append(loss.item())
    return np.mean(val_losses)

#=======================================================================================================================
print('\n[%] Post-training Evaluation')
# Precision-Recall and ROC curves
targets, preds = get_model_predictions(handler_val, network, device)
precision, recall, _ = precision_recall_curve(targets, preds)
roc_auc = roc_curve(targets, preds)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(recall, precision)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.subplot(1, 2, 2)
plt.plot(roc_auc[0], roc_auc[1])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#=======================================================================================================================
# Saving enhanced checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses
}, MODEL_SAVE_PATH)
