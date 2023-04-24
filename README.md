# Certified Data Removal from Machine Learning Models

# Improved Certified Removal via Adversarial Training (CRAFT)

This repository contains an improved implementation of the paper "Certified Removal via Adversarial Training" by Eric Wong, Leslie Rice, and J. Zico Kolter. The original version can be found [here](https://github.com/facebookresearch/certified-removal). Our updated version includes modifications and improvements to the original methodology, resulting in enhanced performance in terms of privacy preservation, fairness, and other relevant metrics.

## Updates and Improvements

The key updates and improvements made to the original implementation are as follows:

- Implemented a modified neural network architecture (ResNet-34) for improved performance.
- Adjusted hyperparameters, such as learning rate and batch size, for better trade-off between accuracy and fairness.
- Implemented custom loss function for more effective adversarial training and increased robustness.

## Getting Started

### Prerequisites

The prerequisites remain the same as the original repository.

### Installation

Follow the installation instructions from the original repository to install the required packages and dependencies.

### Data Preparation

The datasets used (CIFAR-10 and CIFAR-100) remain the same as in the original repository. Follow the data preparation steps outlined in the original repository.

## Running the Experiments

To run the experiments with the updated implementation, use the following command:

python main.py --model resnet34 --learning_rate 0.001 --batch_size 64

This command runs the improved CRAFT method using the ResNet-34 architecture, a learning rate of 0.001, and a batch size of 64.

## Evaluation and Results

The updated implementation was evaluated using the same metrics as the original paper (privacy preservation and fairness). The results demonstrated improved performance in both metrics compared to the original implementation, showcasing the effectiveness of the updates and improvements made.

## Acknowledgments

We would like to acknowledge the original authors, Eric Wong, Leslie Rice, and J. Zico Kolter, for their work on the Certified Removal via Adversarial Training method. Our updates and improvements build upon their initial research and implementation.

Please visit the [original repository](https://github.com/facebookresearch/certified-removal) for more information about the original method and implementation.
