import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class HousingDataProcessor:
    def __init__(self, data_url):
        self.data_url = data_url
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.seed = 42

    def load_and_prepare_data(self):
        # Load the dataset and filter records
        self.df = pd.read_csv(self.data_url)
        self.df = self.df[self.df['ocean_proximity'].isin(['<1H OCEAN', 'INLAND'])]
        self.df = self.df[['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                           'population', 'households', 'median_income', 'median_house_value']]
        
        # Handle missing values in total_bedrooms
        # self.df['total_bedrooms'].fillna(0, inplace=True)  # Option 1: Fill with 0
        self.df['total_bedrooms'].fillna(self.df['total_bedrooms'].mean(), inplace=True)  # Option 2: Fill with mean
        
        # Shuffle the dataset and split into train/val/test sets
        self.X = self.df.drop('median_house_value', axis=1)
        self.y = np.log1p(self.df['median_house_value'])
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(
            self.X, self.y, test_size=0.4, random_state=self.seed)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_temp, self.y_temp, test_size=0.5, random_state=self.seed)

    def train_and_evaluate(self):
        # Train linear regression models with different strategies
        models = {
            'With 0': LinearRegression(),
            'With mean': LinearRegression()
        }

        # Train model with filling missing values with 0
        models['With 0'].fit(self.X_train, self.y_train)
        y_pred_with_0 = models['With 0'].predict(self.X_val)

        # Train model with filling missing values with mean (mean computed on training data)
        self.df['total_bedrooms'].fillna(self.X_train['total_bedrooms'].mean(), inplace=True)
        models['With mean'].fit(self.X_train, self.y_train)
        y_pred_with_mean = models['With mean'].predict(self.X_val)

        # Calculate RMSE for both options
        rmse_with_0 = np.sqrt(mean_squared_error(self.y_val, y_pred_with_0))
        rmse_with_mean = np.sqrt(mean_squared_error(self.y_val, y_pred_with_mean))

        return {
            'With 0': round(rmse_with_0, 2),
            'With mean': round(rmse_with_mean, 2)
        }

    def find_best_regularization(self):
        r_values = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
        best_rmse = float('inf')
        best_r = None

        for r in r_values:
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_r = r

        return round(best_r, 6)

    def evaluate_with_different_seeds(self):
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        rmse_scores = []

        for seed in seeds:
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X, self.y, test_size=0.4, random_state=seed)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=seed)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)

        std_deviation = round(np.std(rmse_scores), 3)
        return std_deviation

    def evaluate_on_test_data(self):
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])
        model = LinearRegression()
        model.fit(X_combined, y_combined)
        y_pred_test = model.predict(self.X_test)
        rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        return round(rmse_test, 2)

if __name__ == "__main__":
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv'
    processor = HousingDataProcessor(data_url)
    processor.load_and_prepare_data()

    # Question 1
    print("Question 1:", "total_bedrooms")

    # Question 2
    median_population = processor.df['population'].median()
    print("Question 2:", median_population)

    # Question 3
    rmse_scores = processor.train_and_evaluate()
    print("Question 3:", rmse_scores)

    # Question 4
    best_r = processor.find_best_regularization()
    print("Question 4:", best_r)

    # Question 5
    std_deviation = processor.evaluate_with_different_seeds()
    print("Question 5:", std_deviation)

    # Question 6
    rmse_test = processor.evaluate_on_test_data()
    print("Question 6:", rmse_test)
