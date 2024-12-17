import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    
    """
    Load the dataset, handle missing values, scale features, and split into train-test sets.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load dataset into a DataFrame
    data = pd.read_csv(file_path)
    
    # Handle missing values by filling with column means
    data.fillna(data.mean(), inplace=True)
    
    # Split the dataset into features (X) and target (y)
    X = data.drop(columns=['target'])  # Adjust 'target' as your label column
    y = data['target']
    
    # Scale features using StandardScaler for normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
