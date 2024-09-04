from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MultiClassifierModel:
    def __init__(self):
        # Define models
        self.models = {
            'RandomForest': RandomForestClassifier(),
            'GradientBoosting': GradientBoostingClassifier(),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'NaiveBayes': GaussianNB()
        }
        self.best_models = {}

    def tune_and_train_models(self, X_train, y_train, param_grids):
        for name, model in self.models.items():
            print(f"Training {name}...")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            self.best_models[name] = grid_search.best_estimator_

    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test, y_pred, average='weighted'),
            }
            print(f"{name} - Accuracy: {results[name]['Accuracy']}, Precision: {results[name]['Precision']}, Recall: {results[name]['Recall']}, F1 Score: {results[name]['F1 Score']}")
        return results
