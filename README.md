This repository contains a deep learning program developed to build an image classification model. Transfer Learning using a pre-trained ResNet-101 model is chosen as the machine learning approach. The training procedure involves a stratified split of the dataset into training and validation sets, followed by data loading using PyTorch's DataLoader. Data augmentation is applied to the training set, and the model architecture is created by unfreezing the last layers of the ResNet model and adding a customized fully connected layer. The AdamW optimizer is used, accompanied by a learning rate adjustment for different parts of the model. Weighted Cross-Entropy is utilized as the loss function to handle class imbalance. Training occurs over multiple epochs with an Early Stopping mechanism to avoid overfitting. The model is trained on a GPU if available, and the entire duration of the training process is logged.

Key Features:

Transfer Learning with a pre-trained ResNet-101 model.
Stratified split of the dataset into training and validation sets.
Data loading using PyTorch's DataLoader.
Data augmentation applied to the training set.
Model architecture created by unfreezing the last layers of the ResNet model and adding a customized fully connected layer.
Utilization of the AdamW optimizer.
Learning rate adjustment for different parts of the model.
Weighted Cross-Entropy used as the loss function to handle class imbalance.
Training over multiple epochs with an Early Stopping mechanism to avoid overfitting.
Training on GPU if available.
Logging of the entire duration of the training process.

Requirements:

Python 3.x
PyTorch
torchvision
numpy
tqdm
