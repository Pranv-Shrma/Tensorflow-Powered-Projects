# Student Exam Performance Predictor - Perceptron Neural Network

## Project Overview

This project implements a simple perceptron neural network to predict whether a student will pass or fail an exam based on their study hours and sleep hours. It uses TensorFlow/Keras for building and training the model.

## Goals

*   Demonstrate a basic neural network implementation.
*   Predict student exam outcomes based on study and sleep patterns.
*   Provide clear, step-by-step calculations for each prediction.

## Implementation

The perceptron model is built using Keras and trained on a custom dataset. The model takes two inputs (study hours and sleep hours) and outputs a probability of passing the exam. The sigmoid activation function is used for binary classification.

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
    docker build -t student-perceptron .
    ```

3.  **Run the Docker container:**

    ```bash
    docker run student-perceptron
    ```

    This will execute the `Simple_perceptron.py` script and display the training process, learned parameters, and test case predictions in your terminal.

## Project Structure

*   `Simple_perceptron.py`: The main Python script containing the perceptron implementation and training logic.
*   `Dockerfile`:  Instructions for building the Docker image.
*   `.dockerignore`: Specifies intentionally untracked files that Docker should ignore.
*   `README.md`: This file, providing project information and instructions.

## Understanding the Output

The script will output the following information:

*   Training data.
*   Model architecture.
*   Compilation settings.
*   Training progress and final loss/accuracy.
*   Learned weights and bias.
*   Step-by-step calculations for test cases, including weighted sum, sigmoid activation, and final prediction.
*   Summary of the perceptron's learning.
