# Neural Network from Scratch with Backpropagation

This project implements a **three-layer feedforward neural network from scratch**, focusing on the fundamentals of **backpropagation** and **gradient descent** without using high-level deep learning libraries.

The goal is to understand how learning dynamics, activation functions, and error propagation influence convergence and performance.

---

## Project Overview

The network consists of:
- Input layer
- One hidden layer
- Output layer

Training is performed using **gradient descent**, with weights updated via **backpropagation of the error**.  
The project compares different activation function configurations and analyzes their impact on learning behavior.

---

## Experiments Conducted

Two main architectures were evaluated:

1. **Sigmoid–Sigmoid Network**
   - Sigmoid activation in both hidden and output layers
   - Slower convergence due to vanishing gradients

2. **ReLU–Sigmoid Network**
   - ReLU activation in the hidden layer
   - Sigmoid activation in the output layer
   - Faster convergence and improved stability

Performance was evaluated using **Mean Squared Error (MSE)** over training iterations.

---

## Evaluation Metric

- **Mean Squared Error (MSE)**  
  Used to measure the difference between predicted outputs and target values during training.

Loss curves show clear differences in convergence speed between activation configurations.

---

## Implementation Details

- Fully implemented **from scratch**
- No automatic differentiation
- Manual computation of:
  - Forward pass
  - Backward pass (gradients)
  - Weight updates
- Emphasis on clarity and educational value

---

## Key Observations

- ReLU in the hidden layer significantly improves convergence speed
- Sigmoid activations are prone to saturation and slower learning
- Proper weight initialization and learning rate selection are critical

---

## Possible Extensions

- Multiple hidden layers
- Different loss functions
- Momentum or adaptive optimizers
- Classification tasks instead of regression

---

## Authors

- **Popa Ștefan-Andrei**
- **Eduard Levinschi**

---

## Documentation

For a detailed explanation of the methodology, derivations, and results, see the accompanying report:
- *Report_Popa_Stefan_Andrei_Levinschi_Eduard.pdf*
