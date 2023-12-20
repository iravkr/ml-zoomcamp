# Cirrhosis Prediction Project

This project aims to predict the severity of Cirrhosis in patients based on various medical features. It uses machine learning models to classify patients into different Cirrhosis stages - 'C', 'CL', or 'D'.

## Problem Statement

Cirrhosis is a liver disease that progresses through different stages, and early detection is crucial for effective treatment. This project focuses on predicting the stage of Cirrhosis in patients using clinical data.

## Data
The data is from Multi-Class Prediction of Cirrhosis Outcomes competition dataset, which can be found in [Kaggle Playground Competition](https://www.kaggle.com/competitions/playground-series-s3e26)




## Feature Extraction

The dataset used in this project contains both categorical and numerical features. The following steps are involved in feature extraction:

1. **Categorical Features**: The categorical features include 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', and 'Status'. These features need to be properly encoded before model training.

2. **Numerical Features**: The numerical features include 'id', 'N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', and 'Stage'. These features are used as-is in the model.

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

## Model Accuracy and Selection

After training and evaluating the models, the best model is selected based on its accuracy. The accuracy of the selected model is reported in the README.md.
- Learning Rate: 0.01
- Max Depth: 7
- Number of Estimators: 400
- Random State: 42
- Subsample: 0.7

## Docker and Pipenv

To create a Docker environment with Pipenv for this project, follow these steps:

1. Install Docker on your system if not already installed.

2. Navigate to the project directory and create a `Dockerfile` with the following content:

```bash
    docker build -t cirrhosis-predict .
```
3. Run the Docker container:
```bash
    docker run -p 9696:9696 cirrhosis-predict
```

Now you can access the Flask API at http://localhost:9696/predict and use it for making predictions as explained in the previous section.
