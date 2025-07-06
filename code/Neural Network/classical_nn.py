# Packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from netcal.metrics import ECE, MCE
from netcal.presentation import ReliabilityDiagram
import torch.distributions as dists
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import random


# Load MNIST Fashion dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Shape and Size of Dataset
print(train_images.shape)
print(test_images.shape)
print(len(train_labels))
print(len(test_labels))

# First Element of Training Data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Normalize the dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# Split into train and validation datasets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Print the first 25 normalized training images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Build the Model
# 1. Flatten Layer: Converts 2D input (28x28) into 1D vector (784,)
# 2. Hidden Dense Layers: 128 and 64 neurons with ReLU activation
# 3. Output Layer: 10 neurons (one for each class)
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation="relu"),
    layers.Dense(10)
])

# Compile the model
# - Optimizer: Adam
# - Loss: SparseCategoricalCrossentropy (used for two or more integer-labeled classes)
# - Metric: Accuracy
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model with training data and 30 epochs
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels),epochs=30,batch_size=64,verbose=2)

# Evaluate the performance of the trained model on the test dataset
# verbose=2 shows one line per test step
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# Add Softmax-layer (convert the linear outputs (logits) of the model to probabilities) for Predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# Make Predictions on the test images
predictions = probability_model.predict(test_images)

# Print the prediction probabilities for the first test image
print(predictions[0]) # Array of 10 probabilities (one per class)
print(np.argmax(predictions[0])) # Index of the class with highest probability
print(test_labels[0]) # Actual label of the first test image

# Function to plot a single image along with predicted and true labels
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# Function to plot the probability distribution for all 10 classes
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Check predictions for a single test image (index 0)
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Check predictions for another test image (index 15)
i = 15
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Visualize multiple test images and their predictions
# Randomly select 9 indices from the test set
# Correct predictions shown in blue, incorrect ones in red
num_rows = 3
num_cols = 3
num_images = num_rows*num_cols
random_indices = random.sample(range(len(test_images)), num_images)
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for idx, i in enumerate(random_indices):
    plt.subplot(num_rows, 2*num_cols, 2*idx+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*idx+2)
    plot_value_array(i, predictions[i], test_labels)
plt.suptitle("NN Predictions on Test Images (Blue = Correct, Red = Incorrect)", fontsize=16)
plt.savefig("nn_predictions_on_test_data", dpi=300)
plt.show()

# Make a prediction for a single image
# Select one image from the test dataset (e.g., index 1)
# img = test_images[1]
# print(img.shape)

# Add a batch dimension since the model expects batches
# img = (np.expand_dims(img,0))
# print(img.shape)

# Predict the class probabilities for this single image
# predictions_single = probability_model.predict(img)
# print(predictions_single)

# Plot the prediction probabilities for this image
# plot_value_array(1, predictions_single[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
# plt.show()

# Print the predicted class index
# print(class_names[np.argmax(predictions_single[0])])

# Plots for the Training and Validation Accuracy and Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.scatter(x=[29], y=[test_acc], color='green', label=f'Test Accuracy: {test_acc:.2f}', zorder=5)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.suptitle("Training and Validation Accuracy and Loss (Classical NN)", fontsize=15)
plt.savefig("nn_acc_loss", dpi=300, bbox_inches='tight')
plt.show()


# Accuracy Values for Training, Test and Validation
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"Training accuracy: {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Calculate Negative Log-Likelihood (NLL) on the test set
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
nll = loss_fn(test_labels, model(test_images, training=False)).numpy()
print(f"Negative Log-Likelihood (NLL): {nll:.4f}")

# Get predicted class labels
pred_probs = predictions
pred_labels = np.argmax(pred_probs, axis=1)

# Compute ECE and MCE using netcal
ece_metric = ECE(bins=15)
mce_metric = MCE(bins=15)

# Measure calibration errors
ece = ece_metric.measure(pred_probs, test_labels)
mce = mce_metric.measure(pred_probs, test_labels)

# Print the results
print(f"Expected Calibration Error (ECE): {ece:.2%}")
print(f"Maximum Calibration Error (MCE): {mce:.2%}")

# Plot the reliability diagram
rd = ReliabilityDiagram(bins=15)
rd.plot(pred_probs, test_labels)
plt.title("Reliability Diagram (NN)")
plt.savefig("nn_reliability_diagram.png", dpi=300)
plt.show()

# Save the trained model
model.save("models/classical_nn.h5")
with open("../../models/classical_nn.json", "w") as f:
    json.dump(history.history, f)
