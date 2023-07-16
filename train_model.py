# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import pickle

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('zomato_clean.csv')

# Split the data into training and testing sets with stratified sampling
train_data, test_data = train_test_split(data, test_size=0.01, random_state=42)

# Save the training and testing sets to separate CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)


# Set unwanted columns
unwanted_columns = ['name', 'type', 'dish_liked']

# Write functions for preprocess data, create pipline and train model

# Add logging statement
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(data, unwanted_columns):
    """
    Preprocess the data.

    Parameters:
    - data (pandas.DataFrame): The data to be preprocessed.
    - unwanted_columns (list): List of unwanted columns to drop.

    Returns:
    - pandas.DataFrame: The preprocessed data.
    """
    # Drop unwanted columns
    data.drop(unwanted_columns, axis=1, inplace=True)

    # Drop null values
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

def create_pipeline():
    """
    Create the pipeline for training an Extra Trees Regressor.

    Returns:
    - sklearn.pipeline.Pipeline: The pipeline for training the model.
    """
    # Define the columns for different transformations
    numeric_columns = ['votes']
    binary_columns = ['online_order', 'book_table']
    categorical_columns = ['location', 'rest_type', 'cuisines']

    # Create pipeline for preprocessing numeric features
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Create pipeline for preprocessing binary features
    binary_transformer = Pipeline([
        ('encoder', OrdinalEncoder())
    ])

    # Create pipeline for preprocessing categorical features
    categorical_transformer = Pipeline([
        ('encoder', OrdinalEncoder())
    ])

    # Combine the transformers using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('numeric_preprocess', numeric_transformer, numeric_columns),
        ('binary_preprocess', binary_transformer, binary_columns),
        ('categorical_preprocess', categorical_transformer, categorical_columns)
    ])

    # Create the final pipeline with preprocessor and Extra Trees Regressor
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', ExtraTreesRegressor(n_estimators=120))
    ])

    return pipeline

def train_model(X, Y):
    """
    Train the Extra Trees Regressor model.

    Parameters:
    - X (pandas.DataFrame): The input features.
    - Y (pandas.Series): The target variable.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - sklearn.pipeline.Pipeline: The trained model.
    """
    # Create the pipeline
    pipeline = create_pipeline()

    # Fit the pipeline to the training data
    pipeline.fit(X, Y)

    return pipeline


# Create and train the model

# Add logging statement
logging.info("Loading and preprocessing data...")

# Load the data
data = pd.read_csv('train_data.csv')

# Splitting data
X = data.drop(['rate'], axis=1)
Y = data['rate']

# Preprocess the data
preprocessed_data = preprocess_data(data, unwanted_columns)

# Train the model
model = train_model(X, Y)

# Save the trained model
pickle.dump(model, open('model.pkl', 'wb'))

# Add logging statement
logging.info("Model trained and saved.")