# Student Grade Predictor - Multi-Layer Neural Network

## Project Overview

This project implements a multi-layer neural network to predict a student's final grade (A, B, C, D, F) based on their study hours, sleep hours, and assignment score. It uses TensorFlow/Keras for building and training the model.

## Goals

*   Demonstrate a multi-layer neural network implementation.
*   Predict student grades based on study habits and assignment performance.
*   Provide clear, step-by-step calculations for test case predictions, including layer activations and softmax output.

## Implementation

The model consists of an input layer, two hidden layers with ReLU activation, and an output layer with softmax activation for multi-class classification.  It is trained on a custom dataset.

## Running the Project

### Prerequisites

*   Docker

### Steps

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Build the Docker image:**

    ```bash
    docker build -t student-grade-predictor .
    ```

3.  **Run the Docker container:**

    ```bash
    docker run student-grade-predictor
    ```

    This will execute the `small_neural_net.py` script and display the training process, model architecture, and test case predictions in your terminal.

## Project Structure

*   `small_neural_net.py`: The main Python script containing the multi-layer neural network implementation and training logic.
*   `Dockerfile`:  Instructions for building the Docker image.
*   `.dockerignore`: Specifies intentionally untracked files that Docker should ignore.
*   `README.md`: This file, providing project information and instructions.

## Understanding the Output

The script will output the following information:

*   Training data.
*   Model architecture.
*   Compilation settings.
*   Training progress and final loss/accuracy.
*   Step-by-step calculations for test cases, including layer activations and softmax output.
*   Insights into how multi-layer networks learn better.
