# Efficient Per-Example Gradient Computations

This repository contains an implementation of three gradient computation methods for neural networks:
- Full: Computes the averaged gradient of the complete objective function
- Naive: Computes each individual gradient by repeatedly calling backward
- Goodfellow: Computes the individual gradients using Goodfellow's Trick, which is equivalent to redefining the backward pass to _not_ aggregate individual gradients

The implementation is based on the following paper:
"Efficient Per-Example Gradient Computations" by Dami Choi, Tamas Sarlos, and Levent Sagun, ICML 2019.

## Requirements
- Python 3
- PyTorch

## Usage

### Run the Code
To run the code, simply execute the `main.py` file. You can modify the `setups` list at the bottom of the file to experiment with different model parameters.

### Output
The output of the program includes:
- Checking correctness: This checks the correctness of the computed gradients by comparing them to the full batch gradient using torch.norm().
- Simple timing: This measures the time taken to compute the gradients using each method for the given model.
- Profiling: This measures the time spent in different functions in the code using the python `cProfile` module.

### Update
In order to upgrade the methodology, the number of hidden layers in the MLP model can be increased to explore the impact of model depth on performance. Additionally, the batch size `N` can also be modified to see the impact of larger or smaller batch sizes on performance. These changes can be made in the `helpers.py` file.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

