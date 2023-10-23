import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from itertools import product
import xgboost as xgb
from xgboost import DMatrix
from math import sqrt



class housingPredictionTree:

    def __init__(self, data_url):
        self.data_url = data_url
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.seed = 1

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.data_url)
        self.df = self.df[self.df['ocean_proximity'].isin(['<1H OCEAN', 'INLAND'])]
        self.df = self.df.fillna(0)
        self.X = self.df.drop('median_house_value', axis=1)
        self.y = np.log1p(self.df['median_house_value'])
        self.X_train_all, self.df_test, self.y_train_all, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.seed)
        self.df_train, self.df_val, self.y_train, self.y_val = train_test_split(self.X_train_all, self.y_train_all, test_size=0.25, random_state=self.seed)
        dv = DictVectorizer(sparse=False)
        train_dict = self.df_train.to_dict(orient='records')
        self.X_train = dv.fit_transform(train_dict)
        val_dict = self.df_val.to_dict(orient='records')
        self.X_val = dv.transform(val_dict)
        test_dict = self.df_test.to_dict(orient='records')
        self.X_test = dv.transform(test_dict)

    def train_and_evaluate(self):
        dt = DecisionTreeRegressor(max_depth=1)
        dt.fit(self.X_train, self.y_train)
        y_pred_train = dt.predict(self.X_train)
        y_pred_val = dt.predict(self.X_val)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_val = mean_squared_error(self.y_val, y_pred_val)
        print('Train MSE:', mse_train)
        print('Validation MSE:', mse_val)
        # Get the decision tree structure
        tree_text = export_text(dt, feature_names=self.df.columns.tolist())
        print(tree_text)

    def train_random_forest(self):
        rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        y_pred_val = rf.predict(self.X_val)
        rmse_val = sqrt(mean_squared_error(self.y_val, y_pred_val))
        print('Validation RMSE:', rmse_val)

    def experiment_with_n_estimators(self, max_estimators=200, step=10):
        best_n_estimators = None
        best_rmse = float('inf')
        for n_estimators in range(10, max_estimators + 1, step):
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=1, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            y_pred_val = rf.predict(self.X_val)
            rmse_val = sqrt(mean_squared_error(self.y_val, y_pred_val))
            print(f'n_estimators={n_estimators}, Validation RMSE: {rmse_val:.3f}')
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_n_estimators = n_estimators
            else:
                break  # Stop if RMSE stops improving
        print(f"RMSE stopped improving after n_estimators={best_n_estimators}")

    def select_best_max_depth(self, max_depth_values=[10, 15, 20, 25], n_estimators_range=range(10, 201, 10)):
        best_max_depth = None
        best_n_estimators = None
        best_mean_rmse = float('inf')
        for max_depth, n_estimators in product(max_depth_values, n_estimators_range):
            rf = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=1, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            y_pred_val = rf.predict(self.X_val)
            rmse_val = sqrt(mean_squared_error(self.y_val, y_pred_val))
            if rmse_val < best_mean_rmse:
                best_mean_rmse = rmse_val
                best_max_depth = max_depth
                best_n_estimators = n_estimators
        print(f"Best max_depth: {best_max_depth}")
        print(f"Best n_estimators: {best_n_estimators}")
        print(f"Best mean RMSE: {best_mean_rmse:.3f}")
    def find_most_important_feature(self):
        rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)

        # Get feature importances
        feature_importances = rf.feature_importances_
        feature_names = self.df.columns[:-1]  # Exclude the target column

        # Create a dictionary to map feature names to importances
        feature_importance_dict = dict(zip(feature_names, feature_importances))

        # Find the most important feature
        most_important_feature = max(feature_importance_dict, key=feature_importance_dict.get)
        
        print("Most Important Feature:", most_important_feature) 

    def train_xgboost(self, eta_values=[0.3, 0.1]):
        xgb_params = {
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'reg:squarederror',
            'nthread': 8,
            'seed': 1,
            'verbosity': 1,
        }

        for eta in eta_values:
            xgb_params['eta'] = eta

            # Create DMatrix for train and validation
            dtrain = DMatrix(self.X_train, label=self.y_train)
            dval = DMatrix(self.X_val, label=self.y_val)

            # Create a watchlist
            watchlist = [(dtrain, 'train'), (dval, 'val')]

            # Train the XGBoost model for 100 rounds
            num_round = 100
            bst = xgb.train(xgb_params, dtrain, num_round, watchlist)

            # Get predictions on the validation set
            y_pred_val = bst.predict(dval)

            # Calculate RMSE
            rmse_val = sqrt(mean_squared_error(self.y_val, y_pred_val))
            print(f'eta={eta}, Validation RMSE: {rmse_val:.3f}')




if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"

    housing_tree = housingPredictionTree(data_url)
    housing_tree.load_and_prepare_data()
    housing_tree.train_and_evaluate()
    housing_tree.train_random_forest()
    housing_tree.experiment_with_n_estimators()
    housing_tree.select_best_max_depth()
    housing_tree.find_most_important_feature()
    housing_tree.train_xgboost()




