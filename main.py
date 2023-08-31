import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize random weights for input to hidden layer
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)

        # Initialize random weights for hidden to output layer
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def forward(self, input_data):
        # Calculate values in the hidden layer
        hidden_output = sigmoid(np.dot(input_data, self.weights_input_hidden))
        
        # Calculate final output
        final_output = sigmoid(np.dot(hidden_output, self.weights_hidden_output))
        
        return final_output

# Create a neural network with 2 input nodes, 2 hidden nodes, and 1 output node
neural_network = NeuralNetwork(2, 2, 1)

# Generate random input
input_data = np.random.rand(2)

# Perform forward pass through the neural network
output = neural_network.forward(input_data)

# Display input and output
print("Input:", input_data)
print("Output:", output)
