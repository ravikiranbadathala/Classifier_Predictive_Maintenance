import pandas as pd
from sklearn.model_selection import train_test_split
from multi_classifier_model import MultiClassifierModel
from data_cleaner_feature_selector import DataCleanerFeatureSelector

def main():
    # Load your dataset
    df = pd.read_csv('truck_maintenance_data.csv')

    # Split data into features and target
    X = df.drop('target_column', axis=1)
    y = df['target_column']

    # Initialize the DataCleanerFeatureSelector class
    data_cleaner = DataCleanerFeatureSelector()

    # Data cleaning and feature selection
    X_clean = data_cleaner.remove_constant_variance(X)
    X_selected = data_cleaner.select_k_best_features(X_clean, y, k=10)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Initialize the MultiClassifierModel class
    model_trainer = MultiClassifierModel()

    # Define hyperparameters for each model
    param_grids = {
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
        'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'KNN': {'n_neighbors': [3, 5, 7]},
        'NaiveBayes': {}  # No hyperparameters for Naive Bayes
    }

    # Train and tune models
    model_trainer.tune_and_train_models(X_train, y_train, param_grids)

    # Evaluate models on the test set
    results = model_trainer.evaluate_models(X_test, y_test)

    # Print final results
    print("Final Results:", results)

if __name__ == "__main__":
    main()
