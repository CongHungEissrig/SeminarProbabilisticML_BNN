# Packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from netcal.metrics import ECE
import torch.distributions as dists
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from netcal.metrics import ECE, MCE
from netcal.presentation import ReliabilityDiagram
import random

# Set seed
np.random.seed(42)
tf.random.set_seed(42)

# Load Fashion-MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Split into train and validation datasets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42)

# Learning rate schedule
batch_size = 64
steps_per_epoch = train_images.shape[0] // batch_size
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=steps_per_epoch * 1000,
    decay_rate=1,
    staircase=False
)

# KL divergence function (scaled per batch)
kl_div_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(train_images.shape[0], tf.float32)

# BNN model with DenseFlipout (Variational Inference)
model_bnn = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    tfp.layers.DenseFlipout(128, activation='relu', kernel_divergence_fn=kl_div_fn),
    layers.Dropout(0.3),
    tfp.layers.DenseFlipout(64, activation='relu', kernel_divergence_fn=kl_div_fn),
    layers.Dropout(0.3),
    tfp.layers.DenseFlipout(10, kernel_divergence_fn=kl_div_fn)
])

# Compile BNN
model_bnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train BNN
history_bnn = model_bnn.fit(
    train_images, train_labels,
    epochs=30,
    batch_size=batch_size,
    validation_data=(val_images, val_labels)
)

# Evaluate on test set
test_loss, test_acc = model_bnn.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy (BNN):", test_acc)


# Predictions + uncertainties via Monte Carlo sampling
def predict_bnn_mc_with_uncertainty(model, images, num_samples=10):
    preds = np.stack([tf.nn.softmax(model(images)).numpy() for _ in range(num_samples)], axis=0)
    mean_preds = preds.mean(axis=0)
    epist_unc = preds.var(axis=0, ddof=1)
    alea_unc = np.mean(preds * (1 - preds), axis=0)
    total_unc = alea_unc + epist_unc
    return mean_preds, total_unc, alea_unc, epist_unc


# Apply on test data
mean_preds, total_unc, alea_unc, epist_unc = predict_bnn_mc_with_uncertainty(model_bnn, test_images, num_samples=50)

# Plotting functions
def plot_image(i, predictions_array, epistunc_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    confidence = 100 * np.max(predictions_array)
    epistunc_mean = np.mean(epistunc_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} {confidence:.1f}% | EU: {epistunc_mean:.4f}\n"
               f"True: {class_names[true_label]}",
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    bars = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    bars[predicted_label].set_color('red')
    bars[true_label].set_color('blue')


# Example plot
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, mean_preds[i], epist_unc[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, mean_preds[i], test_labels)
plt.show()

# Accuracy & Loss plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_bnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_bnn.history['val_accuracy'], label='Validation Accuracy')
plt.scatter(x=[29], y=[test_acc], color='green', label=f'Test Accuracy: {test_acc:.2f}', zorder=5)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_bnn.history['loss'], label='Train Loss')
plt.plot(history_bnn.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.suptitle("Training and Validation Accuracy and Loss (BNN-VI)", fontsize=15)
#plt.savefig("bnn_vi_acc_loss", dpi=300, bbox_inches='tight')
plt.show()

# Accuracy Values for Training, Test and Validation
train_acc_bnn = history_bnn.history['accuracy'][-1]
val_acc_bnn = history_bnn.history['val_accuracy'][-1]

print(f"Training accuracy (BNN): {train_acc_bnn:.4f}")
print(f"Validation accuracy (BNN): {val_acc_bnn:.4f}")
print(f"Test accuracy (BNN): {test_acc:.4f}")

# Visualize multiple test images
num_rows = 3
num_cols = 3
num_images = num_rows * num_cols

random_indices = random.sample(range(len(test_images)), num_images)
plt.figure(figsize=(2.8 * 2 * num_cols, 2.8 * num_rows))

for idx, i in enumerate(random_indices):
    plt.subplot(num_rows, 2 * num_cols, 2 * idx + 1)
    plot_image(i, mean_preds[i], total_unc[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * idx + 2)
    plot_value_array(i, mean_preds[i], test_labels)

plt.subplots_adjust(wspace=1.0, hspace=1.0)
plt.suptitle("BNN VI Predictions on Test Images (Blue = Correct, Red = Incorrect)", fontsize=16)
#plt.savefig("bnn_vi_predictions_on_test_data.png", dpi=300)
plt.show()

#  Calculate Negative Log-Likelihood (NLL) on the test set
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
nll = loss_fn(test_labels, model_bnn(test_images, training=False)).numpy()
print(f"Negative Log-Likelihood (NLL): {nll:.4f}")

# Compute ECE and MCE using netcal
ece = ECE(bins=15)  # Bins kannst du anpassen
mce = MCE(bins=15)

# Measure calibration errors
ece_score = ece.measure(mean_preds, test_labels)
mce_score = mce.measure(mean_preds, test_labels)

print(f"Expected Calibration Error (ECE): {ece_score:.2%}")
print(f"Maximum Calibration Error (MCE): {mce_score:.2%}")

# Plot Reliability Diagram
rel_diag = ReliabilityDiagram(bins=15)
rel_diag.plot(mean_preds, test_labels, show_bars=False)
plt.title("Reliability Diagram (BNN-VI)")
#plt.savefig("bnn_vi_reliability_diagram.png", dpi=300)
plt.show()

# Save model
#model_bnn.save('models/bnn_vi.h5')
#model_json = model_bnn.to_json()
#with open('../../models/bnn_vi.json', 'w') as json_file:
#    json_file.write(model_json)
