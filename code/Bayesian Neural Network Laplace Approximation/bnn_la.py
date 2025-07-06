import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from laplace import Laplace
from netcal.metrics import ECE, MCE
from netcal.presentation import ReliabilityDiagram
import torch.distributions as dists
import numpy as np
import matplotlib.pyplot as plt
import random

# Setup
device = torch.device("cpu")
batch_size = 64
learning_rate = 1e-3
epochs = 30

# Data transformations: convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load FashionMNIST training dataset and split into train and validation sets
full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Classical NN-Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

# Initialize model, loss function and optimizer
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and Test Loops
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct

# Training Process
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    train_loop(train_loader, model, loss_fn, optimizer)
    train_loss, train_acc = test_loop(train_loader, model, loss_fn)
    val_loss, val_acc = test_loop(val_loader, model, loss_fn)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Predict function supporting both MAP and Laplace sampling
@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []
    for x, _ in dataloader:
        x = x.to(device)
        # Monte Carlo sampling: Drawing multiple samples from the Laplace-approximated
        # posterior distribution to obtain probabilistic predictions
        if laplace:
            probs = model(x, link_approx='mc', n_samples=50)
        else:
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        py.append(probs)
    return torch.cat(py)

# Evaluate predictions with accuracy, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Negative Log Likelihood (NLL)
def evaluate_probs(probs, targets):
    acc = (probs.argmax(-1) == targets).float().mean().item()
    ece = ECE(bins=15).measure(probs.cpu().numpy(), targets.cpu().numpy())
    mce = MCE(bins=15).measure(probs.cpu().numpy(), targets.cpu().numpy())
    nll = -dists.Categorical(probs).log_prob(targets).mean().item()
    return acc, ece, mce, nll

# Fit Laplace approximation to the last layer's weights using a diagonal Hessian (diagonal Gaussian)
print("\nFitting Laplace approximation...")
la = Laplace(model, "classification", subset_of_weights="last_layer", hessian_structure="diag")
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")

targets_test = test_dataset.targets.to(device)

# Evaluate standard MAP predictions
probs_map = predict(test_loader, model, laplace=False)
acc_map, ece_map, mce_map, nll_map = evaluate_probs(probs_map, targets_test)

# Evaluate Laplace posterior predictive predictions
probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace, ece_laplace, mce_laplace, nll_laplace = evaluate_probs(probs_laplace, targets_test)

print(f"\n[MAP]     Accuracy: {acc_map:.2%}, ECE: {ece_map:.2%}, MCE: {mce_map:.2%}, NLL: {nll_map:.3f}")
print(f"[Laplace] Accuracy: {acc_laplace:.2%}, ECE: {ece_laplace:.2%}, MCE: {mce_laplace:.2%}, NLL: {nll_laplace:.3f}")

# Accuracy & Loss plots
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.scatter(epochs-1, acc_laplace, label="Laplace Accuracy", color='green', s=80)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")
plt.suptitle("Training and Validation Accuracy and Loss (BNN-LA)", fontsize=15)
#plt.savefig("bnn_la_acc_loss", dpi=300, bbox_inches='tight')
plt.show()

# Plot reliability diagrams to visualize calibration of predictions
rd = ReliabilityDiagram(bins=15)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# rd.plot(probs_map.cpu().numpy(), targets_test.cpu().numpy())
# plt.title("MAP Model Reliability")
# plt.subplot(1, 2, 2)

rd.plot(probs_laplace.cpu().numpy(), targets_test.cpu().numpy())
plt.title("Reliability Diagramm (BNN-LA)")
#plt.savefig("bnn_la_reliability_diagram.png", dpi=300)
plt.tight_layout()
plt.show()

# Predictive entropy as a measure of uncertainty from the predictive distribution
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predictive_entropy(probs):
    # Computes predictive entropy as a measure of uncertainty in the predictions
    # based on the probability distributions obtained via MC sampling
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

def plot_image(idx, probs, uncertainty, true_labels, images, class_names):
    img = images[idx].squeeze()
    pred_label = probs.argmax().item()
    true_label = true_labels[idx].item()
    pred_class_name = class_names[pred_label]
    true_class_name = class_names[true_label]

    pred_prob = probs[pred_label].item() * 100  # Vorhersagewahrscheinlichkeit in %

    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    color = 'blue' if pred_label == true_label else 'red'
    plt.title(f"Pred: {pred_class_name} ({pred_prob:.1f}%)\nEntropy: {uncertainty:.3f}\nTrue: {true_class_name}", color=color)


def plot_value_array(idx, probs, true_labels):
    pred_label = probs.argmax().item()
    true_label = true_labels[idx].item()
    plt.bar(range(10), probs.cpu().numpy(), color="#777777")
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.grid(False)
    bar_colors = ['blue' if i == true_label else 'red' if i == pred_label and pred_label != true_label else '#777777' for i in range(10)]
    for i in range(10):
        plt.bar(i, probs[i].cpu().numpy(), color=bar_colors[i])

# Prepare visualization of sample test images with predictions and uncertainties
num_rows, num_cols = 3, 3
num_images = num_rows * num_cols
test_images = test_dataset.data
test_labels = test_dataset.targets
mean_preds = probs_laplace  # Predictions from Laplace approximation
uncertainties = predictive_entropy(mean_preds)
random_indices = random.sample(range(len(test_images)), num_images)
plt.figure(figsize=(2.8 * 2 * num_cols, 2.8 * num_rows))
for idx, i in enumerate(random_indices):
    plt.subplot(num_rows, 2 * num_cols, 2 * idx + 1)
    plot_image(i, mean_preds[i], uncertainties[i].item(), test_labels, test_images, class_names)
    plt.subplot(num_rows, 2 * num_cols, 2 * idx + 2)
    plot_value_array(i, mean_preds[i], test_labels)
plt.subplots_adjust(wspace=1.0, hspace=1.0)
plt.suptitle("BNN LA Predictions on Test Images (Blue = Correct, Red = Incorrect)", fontsize=16)
#plt.savefig("bnn_la_predictions_on_test_data.png", dpi=300)
plt.show()

# MAP-Werte
final_train_acc = train_accuracies[-1]
final_val_acc = val_accuracies[-1]
test_acc = acc_map
print(f"Training accuracy: {final_train_acc:.4f}")
print(f"Validation accuracy: {final_val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Negative Log-Likelihood (NLL): {nll_map:.4f}")
print(f"Expected Calibration Error (ECE): {ece_map*100:.2f}%")
print(f"Maximum Calibration Error (MCE): {mce_map*100:.2f}%")

# Save model weights
#torch.save(model.state_dict(), '../../models/nn_for_la.pth')

