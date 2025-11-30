# LSTM & GRU Next Word Prediction App (Shakespeare Hamlet)

## Overview

This project implements a next word prediction model using an LSTM (Long Short-Term Memory) neural network. The model is trained on the text of Shakespeare's Hamlet and deployed as a Streamlit web application, allowing users to input a sequence of words and receive a prediction for the subsequent word.

## Files

*   **`app.py`**: The main Streamlit application file. This file contains the code for loading the model and tokenizer, defining the prediction function, and creating the user interface.
*   **`LSTM_Next_word.h5`**: The pre-trained LSTM model file.
*   **`tokenizer.pickle`**: The tokenizer file, which maps words to numerical indices.
*   **`experiment.ipynb`**: A Jupyter Notebook containing the code for data collection, preprocessing, model training, and evaluation.
*   **`hamlet.txt`**: The text data set.
*   **`Dockerfile`**: The Dockerfile used to build the Docker image.
*   **`requirements.txt`**: The file listing the Python package dependencies.

## Training Procedure

The LSTM model was trained using the following steps, as detailed in `experiment.ipynb`:

1.  **Data Collection:** The text of Shakespeare's Hamlet was obtained from the `nltk.corpus.gutenberg` dataset and saved to a `hamlet.txt` file.

2.  **Data Preprocessing:** The text was preprocessed by converting it to lowercase and tokenizing it using a `Tokenizer` object. Input sequences were created by splitting the text into overlapping n-grams, which were then padded to ensure uniform length. The data was split into training and testing sets, and the labels (next words) were converted to a categorical representation using one-hot encoding.

3.  **Model Training:** An LSTM model was created using `tensorflow.keras.models.Sequential`. The model consisted of an embedding layer, two LSTM layers, a dropout layer, and a dense output layer with a softmax activation function. The model was compiled using the `categorical_crossentropy` loss function, the `adam` optimizer, and the `accuracy` metric. The model was trained on the training data for 50 epochs. The trained model and tokenizer were saved to files named `LSTM_Next_word.h5` and `tokenizer.pickle`, respectively.

## Using Docker

This section guides you through running the application using Docker.

### Prerequisites

*   Docker installed on your system.

### Steps

1.  **Build the Docker Image:**

    Open a terminal in the directory containing the `Dockerfile` (where your project files are located) and run the following command:

    ```bash
    docker build -t lstm-next-word-app .
    ```

    This command builds a Docker image named `lstm-next-word-app`. The `.` at the end of the command specifies that the build context is the current directory.

2.  **Run the Docker Container:**

    Run the Docker image with the following command:

    ```bash
    docker run -p 80:80 lstm-next-word-app
    ```

    This command starts a Docker container based on the `lstm-next-word-app` image. The `-p 80:80` flag maps port 80 on your host machine to port 80 inside the container, where the Streamlit app is running.

### Access the Application

Open your web browser and navigate to `http://localhost` or `http://127.0.0.1` to access the application.