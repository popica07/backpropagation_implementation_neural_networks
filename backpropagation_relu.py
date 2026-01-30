import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):  # activation function used, is the same like in the lab
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):  # this is the derivate of sigmoid function
    return a * (1 - a)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

alfa = 0.8 # hint that a bigger one is better in our case (specific for our example)
MSE_values = []
iterations = []

X = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],  # input layer
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])

Y = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],  # output which is similar to input
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])

rng = np.random.default_rng(0)
# W_input_hidden = rng.uniform(-0.4, 0.1, (8, 3))  # 8 inputs → 3 hidden neurons
# W_hidden_output = rng.uniform(-0.4, 0.1, (3, 8))  # 3 hidden neurons → 8 outputs

W_input_hidden = np.array([
    [0.1, 0.5, 0.1],
    [0.2, 0.4, 0.1],
    [0.3, 0.3, 0.1],
    [0.4, 0.2, 0.1],   # first weights which was chosen randomly
    [0.5, 0.1, 0.1],
    [0.6, 0.0, 0.1],
    [0.7, -0.1, 0.1],
    [0.8, -0.2, 0.1]
])

W_hidden_output = np.array([
    [0.1, 0.5, 0.1, 0.3, 0.4, 0.2, 0.4, 0.5],
    [0.1, 0.3, 0.4, 0.5, 0.2, 0.3, 0.2, 0.3],  # second part of weights
    [0.1, 0.4, 0.4, 0.1, 0.3, 0.4, 0.2, 0.3]
])

biases_hidden = np.array([[1, 1, 1]])
biases_output = np.array([[1, 1, 1, 1, 1, 1, 1, 1]])

for i in range(1, 10000):  # loop from 1 to 10000 , where forward and backpropagation is going
    print(f"iteration number: {i}")

    Z_hidden = X @ W_input_hidden + biases_hidden
    A_hidden = relu(Z_hidden)  # basically forward propagation

    Z_output = A_hidden @ W_hidden_output + biases_output
    A_output = sigmoid(Z_output)

    MSE = 0.5 * np.mean((A_output - Y) ** 2)  # computing the cost function (MSE)
    MSE_values.append(MSE)
    iterations.append(i)
    print(MSE)

    if MSE < 0.05:
        print("we stopped the machine at iteration number:", i)
        print("MSE value is:", MSE)
        break

    delta_Z_hidden_output = (A_output - Y) * sigmoid_derivative(A_output)

    gradient_W_hidden_output = (A_hidden.T @ delta_Z_hidden_output) / 8  # backpropagation for the hidden-output layer
    gradient_biases_output = np.sum(delta_Z_hidden_output, axis=0, keepdims=True) / 8

    delta_Z_input_hidden = (delta_Z_hidden_output @ W_hidden_output.T) * relu_derivative(A_hidden)

    gradient_W_input_hidden = (X.T @ delta_Z_input_hidden) / 8  # backpropagation for the input-hidden layer
    gradient_biases_hidden = np.sum(delta_Z_input_hidden, axis=0, keepdims=True) / 8

    W_input_hidden = W_input_hidden - alfa * gradient_W_input_hidden
    W_hidden_output = W_hidden_output - alfa * gradient_W_hidden_output
    biases_hidden = biases_hidden - alfa * gradient_biases_hidden
    biases_output = biases_output - alfa * gradient_biases_output

plt.plot(iterations, MSE_values, label=f"α = {alfa}")
plt.title("MSE evolution over iterations with ReLU ")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()
