from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils.data_preprocessing import load_and_preprocess_data
from utils.evaluation import evaluate_model

def run_traditional_ml(data_path):
    """
    Train and evaluate traditional ML models: Logistic Regression and Random Forest.

    Args:
        data_path (str): Path to the dataset.
    """
    # Preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Train Logistic Regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Evaluate and print performance metrics
    print("Logistic Regression Metrics:")
    print(evaluate_model(y_test, y_pred_lr))
    
    print("\nRandom Forest Metrics:")
    print(evaluate_model(y_test, y_pred_rf))
