# CNN Quantization in PyTorch

## Overview
This project explores Convolutional Neural Network (CNN) performance and optimization through model quantization techniques in PyTorch. The implementation demonstrates how to reduce model size while maintaining accuracy, a crucial skill for deploying machine learning models in resource-constrained environments.

## Key Features
- Implementation of custom CNN architectures for image classification on CIFAR-10
- Post-training quantization of models to reduce size by ~70% with minimal accuracy impact
- Transfer learning with pre-trained models (ResNet50, VGG16) for image classification
- Fine-tuning techniques on the Hymenoptera dataset (ants vs. bees classification)
- Comparative analysis of model architectures, dropout regularization, and quantization effects

## Technologies
- PyTorch
- torchvision
- Matplotlib for visualization
- Pre-trained models (ResNet50, VGG16, ResNet18)

## Project Structure
The project is organized into four main exercises:

### 1. CNN Implementation on CIFAR-10
- Built a baseline CNN model achieving 62% accuracy
- Improved architecture with additional convolutional layers and dropout regularization
- Enhanced model achieved 74% accuracy with better generalization

### 2. Model Quantization
- Applied post-training dynamic quantization to reduce model sizes by approximately 70%
- Demonstrated negligible accuracy impact despite significant size reduction
- Implemented quantization-aware training (QAT) for potentially better performance

### 3. Pre-trained Model Classification
- Used ResNet50 and VGG16 pre-trained on ImageNet for image classification
- Applied quantization to these models and evaluated performance
- Achieved 75% model size reduction while maintaining classification accuracy

### 4. Transfer Learning
- Fine-tuned ResNet18 on the Hymenoptera dataset (ants vs. bees)
- Experimented with custom classification layers and dropout
- Improved accuracy from 92% to 98% with advanced network head design
- Demonstrated quantization techniques on transfer learning models

## Results
- CNN model quantization reduced size by ~70% with minimal accuracy impact
- Transfer learning with customized network head achieved 98% accuracy on binary classification
- Quantized models demonstrated excellent performance-to-size ratio
- Dropout regularization proved effective at preventing overfitting

## Usage
To run the notebook:

1. Install the required dependencies:
```bash
pip install torch torchvision matplotlib numpy pillow
```

Stevan Le Stanc