# Energy Consumption Dataset

## Introduction
This dataset provides a comprehensive look at various factors influencing energy consumption in a given environment. It covers a range of parameters including temperature, humidity, square footage of the space, occupancy levels, usage of heating, ventilation, and air conditioning (HVAC), lighting usage, renewable energy generation, and more. The data is timestamped and includes additional context such as the day of the week and holiday occurrences.

## Dataset Description
The dataset is structured in a tabular format where each row represents data for a specific hour. The data columns provide detailed insights into environmental conditions, building characteristics, and energy usage.

### Columns Description:
1. **Timestamp:** Date and time of the data entry, in the format `YYYY-MM-DD HH:MM:SS`.
2. **Temperature:** Ambient temperature at the time of data entry, measured in degrees Celsius.
3. **Humidity:** Relative humidity percentage at the time of data entry.
4. **SquareFootage:** Total area of the space in square feet.
5. **Occupancy:** Number of occupants in the space at the time of data entry.
6. **HVACUsage:** Indicates whether the Heating, Ventilation, and Air Conditioning (HVAC) system was on or off.
7. **LightingUsage:** Indicates whether the lights were on or off.
8. **RenewableEnergy:** Amount of renewable energy generated at the time of data entry (in some units).
9. **DayOfWeek:** Day of the week corresponding to the timestamp.
10. **Holiday:** Indicates whether the day is a holiday (`Yes` or `No`).
11. **EnergyConsumption:** Total energy consumption at the time of data entry (in some units).

## Usage
This dataset can be used for various analytical purposes, such as understanding energy consumption patterns, identifying the impact of different factors on energy usage, and developing predictive models for energy management. It is particularly useful for research in building energy efficiency, smart grid optimization, and environmental studies.

## Notes
- The dataset assumes a consistent geographical location and environmental setting.
- Ensure appropriate data handling and privacy measures are in place if using real-world data.


## Instructions to Run

Follow these steps to replicate the project:

### Feature Extraction

1. Ensure you have the required Python libraries installed.
2. Prepare the dataset and perform necessary preprocessing.
3. Encode categorical features and scale numerical features.

### Model Training

1. Split the data into training, validation, and test sets.
2. Train different models, such as Logistic Regression, Decision Tree, and XGBoost, with hyperparameter tuning.
3. Evaluate and compare the models' accuracy using appropriate metrics.
4. Select the best model based on performance.

### Running Predictions using Flask

1. Create a Flask application (`predict.py`) that loads the best-trained model.
2. Define an endpoint for making predictions.
3. Prepare a customer data dictionary with the required features.
4. Send a POST request to the Flask server with the customer data to get predictions.


## Docker and Pipenv

To create a Docker environment with Pipenv for this project, follow these steps:

1. Install Docker on your system if not already installed.

2. Navigate to the project directory and create a `Dockerfile` with the following content:

```bash
    docker build -t Proj_Capstone_2 .
```
3. Run the Docker container:
```bash
    docker run -p 9696:9696 Proj_Capstone_2
```

Now you can access the Flask API at http://localhost:9696/predict and use it for making predictions as explained in the previous section.
