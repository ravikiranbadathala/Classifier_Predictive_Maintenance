import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

class DataCleanerFeatureSelector:
    def __init__(self):
        pass

    def remove_constant_variance(self, data):
        data = data.select_dtypes(include=[np.number])
        selector = VarianceThreshold(threshold=0)
        selector.fit(data)
        non_constant_columns = selector.get_support(indices=True)
        return data.iloc[:, non_constant_columns]

    def select_k_best_features(self, X, y, k=10):
        selector = SelectKBest(score_func=f_regression, k=k)
        selected_data = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        return pd.DataFrame(selected_data, columns=selected_features)

    def recursive_feature_elimination(self, X, y, n_features_to_select=8):
        model = LinearRegression()
        selector = RFE(model, n_features_to_select=n_features_to_select, step=1)
        selector = selector.fit(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        return X[selected_features]

    def tree_based_feature_selection(self, X, y, n_features=8):
        model = RandomForestRegressor(random_state=123)
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        selected_features = X.columns[indices]
        return X[selected_features]

    def lasso_feature_selection(self, X, y, alpha=0.01):
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        selected_features = X.columns[lasso.coef_ != 0]
        return X[selected_features]

    def plot_correlation_heatmap(self, data):
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={'shrink': .5})
        plt.title('Correlation Heatmap')
        plt.show()
