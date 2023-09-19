import pandas as pd
import numpy as np

class HousingDataAnalyzer:
    def __init__(self, data_url):
        # Load the CSV data into a Pandas DataFrame
        self.data_url = data_url
        self.df = pd.read_csv(data_url)

    def get_pandas_version(self):
        # Question 1: Version of Pandas
        return pd.__version__

    def get_num_columns(self):
        # Question 2: Number of columns in the dataset
        return len(self.df.columns)

    def get_columns_with_missing_values(self):
        # Question 3: Which columns in the dataset have missing values?
        return self.df.columns[self.df.isnull().any()].tolist()

    def get_unique_values_in_ocean_proximity(self):
        # Question 4: Number of unique values in the 'ocean_proximity' column
        return self.df['ocean_proximity'].nunique()

    def get_avg_median_house_value_near_bay(self):
        # Question 5: Average value of the 'median_house_value' for the houses near the bay
        bay_area_houses = self.df[self.df['ocean_proximity'] == 'NEAR BAY']
        return bay_area_houses['median_house_value'].mean()

    def calculate_avg_total_bedrooms(self):
        # Question 6: Calculate the average of the 'total_bedrooms' column
        return self.df['total_bedrooms'].mean()

    def fill_missing_total_bedrooms(self):
        # Use the fillna method to fill missing values in 'total_bedrooms' with the mean value
        mean_total_bedrooms = self.calculate_avg_total_bedrooms()
        self.df['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)

    def calculate_avg_total_bedrooms_after_fill(self):
        # Calculate the average of 'total_bedrooms' again after filling missing values
        return self.df['total_bedrooms'].mean()

    def calculate_last_element_of_w(self):
        # Question 7: Value of the last element of w
        # Perform the operations for w as previously shown
        island_data = self.df[self.df['ocean_proximity'] == 'ISLAND']
        selected_columns = island_data[['housing_median_age', 'total_rooms', 'total_bedrooms']]
        X = selected_columns.to_numpy()
        XTX = np.dot(X.T, X)
        XTX_inverse = np.linalg.inv(XTX)
        y = np.array([950, 1300, 800, 1000, 1300])
        w = np.dot(np.dot(XTX_inverse, X.T), y)
        return w[-1]

# Example usage:
if __name__ == '__main__':
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv'
    analyzer = HousingDataAnalyzer(data_url)

    print("Question 1:", analyzer.get_pandas_version())
    print("Question 2:", analyzer.get_num_columns())
    print("Question 3:", analyzer.get_columns_with_missing_values())
    print("Question 4:", analyzer.get_unique_values_in_ocean_proximity())
    print("Question 5:", analyzer.get_avg_median_house_value_near_bay())

    # Fill missing values and calculate the average again
    analyzer.fill_missing_total_bedrooms()
    print("Question 6 (Before fill):", analyzer.calculate_avg_total_bedrooms())
    print("Question 6 (After fill):", analyzer.calculate_avg_total_bedrooms_after_fill())

    print("Question 7:", analyzer.calculate_last_element_of_w())
