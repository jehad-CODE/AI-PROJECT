# Machine Learning Pipeline for Classification
This project implements and compares the performance of Traditional Machine Learning Models (Logistic Regression and Random Forest) with a Neural Network on a binary classification dataset. The goal is to preprocess data, train the models, evaluate their performance, and analyze the results using common evaluation metrics.

## Project Overview
This project compares the performance of traditional machine learning models (Logistic Regression and Random Forest) with a Neural Network on a binary classification dataset.

## Project Structure
```
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
├── .gitignore                  # Files and directories to ignore in Git
├── requirements.txt            # Python dependencies for the project
├── README.md                   # Documentation 
├── main.py                     # Entry point to run the project
```

## Dataset Information
The dataset used in this project is the **Heart Disease Dataset**, which is intended for binary classification tasks (detecting the presence or absence of heart disease).

### Dataset Structure
- **File Format:** CSV
- **Attributes:**
  - `age`: Age of the patient.
  - `sex`: Gender (1 = Male, 0 = Female).
  - `cp`: Chest pain type (0 to 3).
  - `trestbps`: Resting blood pressure (mm Hg).
  - `chol`: Serum cholesterol (mg/dl).
  - `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
  - `restecg`: Resting ECG results (0 to 2).
  - `thalach`: Maximum heart rate achieved.
  - `exang`: Exercise-induced angina (1 = yes, 0 = no).
  - `oldpeak`: ST depression induced by exercise.
  - `slope`: Slope of the peak exercise ST segment (0 to 2).
  - `ca`: Number of major vessels (0-3) by fluoroscopy.
  - `thal`: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect).
  - `target`: Heart disease presence (1 = disease, 0 = no disease).

### Example Rows
| age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target |
|-----|-----|----|----------|------|-----|---------|---------|-------|---------|-------|----|------|--------|
| 52  | 1   | 0  | 125      | 212  | 0   | 1       | 168     | 0     | 1.0     | 2     | 2  | 3    | 0      |
| 53  | 1   | 0  | 140      | 203  | 1   | 0       | 155     | 1     | 3.1     | 0     | 0  | 3    | 0      |

### Dataset Source
It can be downloaded from the following location:
[UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).



## Code Overview
### 1. Data Preprocessing (utils/data_preprocessing.py)
Handles:
- Loading Data: Reads the CSV file into a Pandas DataFrame.
- Missing Values: Fills missing values with the mean.
- Splitting Data: Splits into 80%-20% train-test sets.
- Feature Scaling: Standardizes features using `StandardScaler`.

### 2. Evaluation Metrics (utils/evaluation.py)
Defines:
- **Accuracy:** Proportion of correct predictions.
- **Precision:** Quality of positive predictions.
- **Recall:** Detection of positive cases.
- **F1 Score:** Harmonic mean of precision and recall.
- **ROC-AUC Score:** Measures separation ability for probabilistic predictions.

### 3. Traditional Machine Learning Models (models/traditional_ml.py)
Implements:
- Logistic Regression: Linear model for binary classification.
- Random Forest: Ensemble model using decision trees.

### 4. Neural Network (models/neural_networks.py)
Implements a feedforward neural network using Keras:
- Input Layer: 16 neurons, ReLU activation.
- Hidden Layer: 8 neurons, ReLU activation.
- Output Layer: 1 neuron, Sigmoid activation.

Training:
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Epochs: 20
- Batch Size: 32

### 5. Main Script (main.py)
Runs:
- Data preprocessing via `data_preprocessing.py`.
- Traditional models via `traditional_ml.py`.
- Neural network via `neural_networks.py`.
- Prints evaluation metrics for all models.

## Running the Project
To run the project without generating `__pycache__` files:
```bash
python -B main.py
```
