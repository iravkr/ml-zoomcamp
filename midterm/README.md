
# Adult Income Dataset - Exploratory Data Analysis (EDA)

## Problem Description

**Dataset**: "adult.csv"

## Dataset

The dataset used in this project is the "Census Income Dataset." You can obtain this dataset from Kaggle.

- [Census Income Dataset on Kaggle](https://www.kaggle.com/datasets/tawfikelmetwally/census-income-dataset/data)

Please download the dataset from the provided link and place it in the appropriate directory to run the code and perform the analysis.

The "Adult Income" dataset is a commonly used dataset in machine learning and data analysis. It contains information about various attributes of individuals, such as their age, workclass, education, marital status, occupation, relationship, race, gender, capital gain, capital loss, hours per week, native country, and income. The dataset is often used for predicting whether an individual's income exceeds $50K per year (the "Income" column) based on the other attributes.


## Problem Statement

The problem at hand is to perform EDA on the "Adult Income" dataset. This involves gaining a deep understanding of the data, its distribution, relationships between variables, and potential insights that can be derived from it. Specifically, we aim to:

1. **Data Loading** 

2. **Data Summary**

3. **Descriptive Statistics**

4. **Missing Values**

5. **Data Distribution**

6. **Categorical Data Distribution**

7. **Correlation Matrix**

8. **Income Distribution by Education Level**

9. **Income Distribution by Gender**

10. **Income Distribution by Race**


## How to Use

- Clone or download this repository to your local machine.

- Ensure you have the necessary libraries and dependencies installed. You can use tools like Jupyter Notebook to run the provided Python code.

- Open the Jupyter Notebook or Python script that contains the code for EDA and feature importance using Mutual Information and Correlation.

- Follow the step-by-step instructions in the code to perform the EDA and feature importance analysis.

## Data Source

The "Adult Income" dataset used in this analysis is available from various sources. In this case, it is provided as "adult.csv."

# Getting Started

To run this project, follow these steps:

## Prerequisites
- **Docker**: You need to have Docker installed on your system.

## Installation

1. Clone this repository.

2. Build the Docker image using the following command:

   ```bash
   docker build -t <image-name> .
   ```

3. Run the Docker container with the following command:

   ```bash
   docker run -it --rm -p 4041:4041 <image-name>
   ```

4. Install development packages with the following command:

   ```bash
   pipenv install --dev
   ```

## Using the Prediction Service

Once you have the container up and running, you can begin using the model by running:

```bash
python test-predict.py
```
