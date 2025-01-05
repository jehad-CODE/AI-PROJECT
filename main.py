from models.traditional_ml import run_traditional_ml
from models.neural_networks import run_neural_network

def main():
    """
    Main function to run the entire pipeline: traditional ML and neural networks.
    """
    data_path = 'data/heart.csv' 
    print("Starting the pipeline with dataset:", data_path)

    # Run Traditional Machine Learning models
    print("\nRunning Traditional Machine Learning models...")
    run_traditional_ml(data_path)

    # Run Neural Network model
    print("\nRunning Neural Network model...")
    run_neural_network(data_path)

if __name__ == "__main__":
    main()
