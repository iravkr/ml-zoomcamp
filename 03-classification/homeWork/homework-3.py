import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class CarPriceAnalyzer:
    def __init__(self, data_url):
        self.data_url = data_url
        self.data = None
        self.features = None
        self.target = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_and_prepare_data(self):
        # Load the dataset
        self.data = pd.read_csv(self.data_url)

        # Fill missing values with 0
        self.data.fillna(0, inplace=True)

        # Rename MSRP variable to price
        self.data.rename(columns={'MSRP': 'price'}, inplace=True)

    def answer_question_1(self):
        # Question 1: What is the most frequent observation (mode) for the column Transmission Type?
        mode_transmission_type = self.data['Transmission Type'].mode()[0]
        return mode_transmission_type

    def answer_question_2(self):
        # Question 2: Calculate the correlation matrix for numerical features
        numerical_features = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg']
        correlation_matrix = self.data[numerical_features].corr()
        
        # Find the two features with the highest correlation
        max_corr = 0
        feature_pair = ()
        for i in range(len(numerical_features)):
            for j in range(i+1, len(numerical_features)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > max_corr:
                    max_corr = abs(corr)
                    feature_pair = (numerical_features[i], numerical_features[j])
        
        return feature_pair

    def make_price_binary(self):
        # Create a binary variable 'above_average' based on price
        mean_price = self.data['price'].mean()
        self.data['above_average'] = (self.data['price'] > mean_price).astype(int)
        self.features = self.data.drop(['price', 'above_average'], axis=1)
        self.target = self.data['above_average']

    def split_data(self):
        # Split data into train/val/test sets (60%/20%/20%)
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(
            self.features, self.target, test_size=0.4, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_temp, self.y_temp, test_size=0.5, random_state=42)

    def answer_question_3(self):
        # Question 3: Calculate mutual information scores between 'above_average' and other categorical variables
        categorical_features = ['Make', 'Model', 'Transmission Type', 'Vehicle Style']
        mutual_info_scores = []
        
        for feature in categorical_features:
            mi_score = mutual_info_score(self.y_train, self.X_train[feature])
            mutual_info_scores.append(round(mi_score, 2))
        
        # Find the variable with the lowest mutual information score
        min_score_variable = categorical_features[np.argmin(mutual_info_scores)]
        
        return min_score_variable

    def train_logistic_regression(self):
        # Perform one-hot encoding on the entire dataset, excluding non-numeric columns
        non_numeric_columns = ['Make', 'Model', 'Transmission Type', 'Vehicle Style', 'Engine Fuel Type',
                               'Driven_Wheels', 'Market Category', 'Vehicle Size']
        self.data_encoded = pd.get_dummies(self.features.drop(non_numeric_columns, axis=1), 
                                           columns=['Number of Doors'], prefix='NumDoors')

        # Split the data into train/val/test sets
        self.X_train_encoded, self.X_temp_encoded, self.y_train_encoded, self.y_temp_encoded = train_test_split(
            self.data_encoded, self.target, test_size=0.4, random_state=42)
        self.X_val_encoded, self.X_test_encoded, self.y_val_encoded, self.y_test_encoded = train_test_split(
            self.X_temp_encoded, self.y_temp_encoded, test_size=0.5, random_state=42)
        
        # Train logistic regression model
        model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
        model.fit(self.X_train_encoded, self.y_train_encoded)
        y_val_pred = model.predict(self.X_val_encoded)
        accuracy = round(accuracy_score(self.y_val_encoded, y_val_pred), 2)
        
        return accuracy

    def feature_elimination(self):
        # Train a logistic regression model with all features
        model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
        model.fit(self.X_train_encoded, self.y_train_encoded)
        base_accuracy = accuracy_score(self.y_val_encoded, model.predict(self.X_val_encoded))

        # Initialize a dictionary to store feature accuracy differences
        feature_accuracy_diff = {}

        # Iterate through features to be removed
        features_to_remove = ['Year']  # Add other features to be removed here
        for feature in features_to_remove:
            reduced_features_train = self.X_train_encoded.drop(feature, axis=1)
            reduced_features_val = self.X_val_encoded.drop(feature, axis=1)
            model.fit(reduced_features_train, self.y_train_encoded)
            reduced_accuracy = accuracy_score(self.y_val_encoded, model.predict(reduced_features_val))
            feature_accuracy_diff[feature] = base_accuracy - reduced_accuracy

        # Find the feature with the smallest difference
        min_diff_feature = min(feature_accuracy_diff, key=feature_accuracy_diff.get)

        return min_diff_feature


    def train_ridge_regression(self):
        # Train Ridge regression with different alpha values
        alphas = [0, 0.01, 0.1, 1, 10]
        best_rmse = float('inf')
        best_alpha = None
        
        for alpha in alphas:
            model = Ridge(alpha=alpha, solver='sag', random_state=42)
            model.fit(self.X_train_encoded, self.y_train_encoded)
            y_val_pred = model.predict(self.X_val_encoded)
            rmse = np.sqrt(mean_squared_error(self.y_val_encoded, y_val_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
        
        return round(best_alpha, 3)

if __name__ == "__main__":
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
    car_analyzer = CarPriceAnalyzer(data_url)
    car_analyzer.load_and_prepare_data()
    
    # Question 1
    ans1 = car_analyzer.answer_question_1()
    print("Question 1:", ans1)
    
    # Question 2
    ans2 = car_analyzer.answer_question_2()
    print("Question 2:", ans2)
    
    # Make price binary and split data
    car_analyzer.make_price_binary()
    car_analyzer.split_data()
    
    # Question 3
    ans3 = car_analyzer.answer_question_3()
    print("Question 3:", ans3)
    
    # Question 4
    ans4 = car_analyzer.train_logistic_regression()
    print("Question 4:", ans4)
    
    # Question 5
    ans5 = car_analyzer.feature_elimination()
    print("Question 5:", ans5)
    
    # Question 6
    ans6 = car_analyzer.train_ridge_regression()
    print("Question 6:", ans6)
