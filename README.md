# Machine Learning Pipeline for Classification
This project implements and compares the performance of Traditional Machine Learning Models (Logistic Regression and Random Forest) with a Neural Network on a binary classification dataset. The goal is to preprocess data, train the models, evaluate their performance, and analyze the results using common evaluation metrics.


## Project Overview
This project compares the performance of traditional machine learning models (Logistic Regression and Random Forest) with a Neural Network on a binary classification dataset.


## Project Structure
AIPROJECT/
│
├── data/                       # Contains the dataset(s)
│   └── heart.csv               # Example dataset for binary classification
│
├── models/                     # Model implementations
│   ├── traditional_ml.py       # Logistic Regression and Random Forest implementation
│   └── neural_networks.py      # Neural Network implementation (Keras/TensorFlow)
│
├── utils/                      # Utility functions
│   ├── data_preprocessing.py   # Data loading, preprocessing, and splitting
│   └── evaluation.py           # Model evaluation metrics
│
├── main.py                     # Entry point to run the project                
└── README.md                   # Documentation (this file)





### Code Overview
1. Data Preprocessing (utils/data_preprocessing.py)
Handles all data-related operations:
Loading Data: Reads a CSV file into a Pandas DataFrame.
Missing Values: Fills missing values with the mean of their respective columns.
Splitting Data: Divides the dataset into training and testing sets (80%-20% split).
Feature Scaling: Standardizes feature values using StandardScaler.

2. Evaluation Metrics (utils/evaluation.py)
Defines performance metrics to evaluate model performance:
Accuracy: Proportion of correct predictions.
Precision: Accuracy of positive predictions.
Recall: Ability to identify all positive cases.
F1 Score: Harmonic mean of Precision and Recall.
ROC-AUC Score: Measures how well a model separates classes (used for probabilistic predictions).

3. Traditional Machine Learning Models (models/traditional_ml.py)
Implements and trains two traditional machine learning models:
Logistic Regression: A simple linear model for binary classification.
Random Forest: An ensemble learning model using multiple decision trees.
Output:
Predictions on the test set.
Evaluation metrics for both models.


4. Neural Network (models/neural_networks.py)
Builds and trains a simple feedforward neural network using Keras (TensorFlow):
Input Layer: 16 neurons with ReLU activation.
Hidden Layer: 8 neurons with ReLU activation.
Output Layer: 1 neuron with Sigmoid activation for binary classification.
Training Details:
Loss: Binary Cross-Entropy
Optimizer: Adam
Epochs: 50
Batch Size: 32
Output:
Predictions on the test set.
Evaluation metrics for the neural network.


5. Main Script (main.py)
The entry point to the project:
Preprocesses the dataset using data_preprocessing.py.
Runs Traditional Machine Learning models using traditional_ml.py.
Runs the Neural Network using neural_networks.py.
Prints evaluation metrics for all models.




#### Explanation of Output:
Accuracy: Measures the proportion of correct predictions.
Precision: Measures the quality of positive predictions.
Recall: Measures how well the model detects positive cases.
F1 Score: Balances Precision and Recall.
ROC-AUC: Used for Neural Networks to measure how well the model separates the classes.