#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Define the sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Define the neural network class
class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;

public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;

        // Initialize random weights for input to hidden layer
        weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                weightsInputHidden[i][j] = (rand() % 1000) / 1000.0; // Random value between 0 and 1
            }
        }

        // Initialize random weights for hidden to output layer
        weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                weightsHiddenOutput[i][j] = (rand() % 1000) / 1000.0; // Random value between 0 and 1
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        // Calculate values in the hidden layer
        std::vector<double> hiddenOutput(hiddenSize, 0.0);
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                hiddenOutput[i] += input[j] * weightsInputHidden[j][i];
            }
            hiddenOutput[i] = sigmoid(hiddenOutput[i]);
        }

        // Calculate final output
        std::vector<double> finalOutput(outputSize, 0.0);
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                finalOutput[i] += hiddenOutput[j] * weightsHiddenOutput[j][i];
            }
            finalOutput[i] = sigmoid(finalOutput[i]);
        }

        return finalOutput;
    }
};

int main() {
    srand(static_cast<unsigned int>(time(nullptr))); // Seed the random number generator

    // Create a neural network with 2 input nodes, 2 hidden nodes, and 1 output node
    NeuralNetwork neuralNetwork(2, 2, 1);

    // Generate random input
    std::vector<double> input = {(rand() % 1000) / 1000.0, (rand() % 1000) / 1000.0};

    // Perform forward pass through the neural network
    std::vector<double> output = neuralNetwork.forward(input);

    // Display input and output
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;

    return 0;
}
