import tensorflow as tf
from utils.data_preprocessing import load_and_preprocess_data
from utils.evaluation import evaluate_model

def run_neural_network(data_path):
    """
    Build, train, and evaluate a simple feedforward neural network.

    Args:
        data_path (str): Path to the dataset.
    """
    # Preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Define a simple feedforward neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Predict on test data
    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
    y_prob = model.predict(X_test).flatten()
    
    # Evaluate the model
    print("Neural Network Metrics:")
    print(evaluate_model(y_test, y_pred, y_prob))
