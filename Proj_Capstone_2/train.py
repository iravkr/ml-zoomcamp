import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

# Reading the data
logging.info("Reading the data...")
energy_data = pd.read_csv('Energy_consumption.csv')

# Dropping the 'Timestamp' column
df = energy_data.drop(columns=['Timestamp'])

# Splitting the data into train, validation, and test sets
logging.info("Splitting the data into train, validation, and test sets...")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.EnergyConsumption.values
y_val = df_val.EnergyConsumption.values
y_test = df_test.EnergyConsumption.values

del df_train['EnergyConsumption']
del df_val['EnergyConsumption']
del df_test['EnergyConsumption']

# Define categorical and numerical features
categorical_features = list(df.dtypes[df.dtypes == 'object'].index)
numerical_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
numerical_features.remove("EnergyConsumption")

# Initialize DictVectorizer
dv = DictVectorizer(sparse=False)

# Transform categorical and numerical features
train_dict = df_train[categorical_features + numerical_features].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical_features + numerical_features].to_dict(orient='records')
X_val = dv.transform(val_dict)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Exporting the model
logging.info("Exporting the model...")
with open('model-reg.bin', 'wb') as f:
    pickle.dump((model, dv), f)

logging.info("Model exported successfully.")
