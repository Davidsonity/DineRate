import argparse
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np
import pickle

# Set unwanted columns
unwanted_columns = ['name', 'url', 'address', 'phone', 'dish_liked', 'reviews_list', 'menu_item', 'listed_in(city)', 'listed_in(type)' ]

def read_data(file_path):
    """
    Read data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pandas.DataFrame: The loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, unwanted_columns, location_encoder, RestType_encoder, cuisines_encoder):
    """
    Preprocess the data.

    Parameters:
    - data (pandas.DataFrame): The data to be preprocessed.
    - unwanted_columns (list): List of unwanted columns to drop.
    - location_encoder (LabelEncoder): The pre-trained label encoder for 'location'.
    - RestType_encoder (LabelEncoder): The pre-trained label encoder for 'rest_type'.
    - cuisines_encoder (LabelEncoder): The pre-trained label encoder for 'cuisines'.

    Returns:
    - pandas.DataFrame: The preprocessed data.
    """
    # Drop unwanted columns
    data.drop(unwanted_columns, axis=1, inplace=True)
    
    # Replace 'NEW' and '-' in column 'rate' with 'NaN'
    data['rate'].replace(['NEW', '-'], np.nan, inplace=True)
    
    # Drop null values
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Rename columns
    data.rename(columns={'approx_cost(for two people)': 'cost_for_2'}, inplace=True)
    
    # Convert columns 'rate' and 'cost_for_2' to float datatype
    data['rate'] = data['rate'].apply(lambda x: x.replace('/5', '') if isinstance(x, str) and '/5' in x else x).astype(float)
    data['cost_for_2'] = data['cost_for_2'].apply(lambda x: x.replace(',', '') if isinstance(x, str) and ',' in x else x).astype(float)
    
    # Convert the online_order categorical variable into numeric format
    data['online_order'] = data['online_order'].map({'Yes': 1, 'No': 0})
    data['online_order'] = pd.to_numeric(data['online_order'])
    
    # Convert the book_table categorical variable into numeric format
    data['book_table'] = data['book_table'].map({'Yes': 1, 'No': 0})
    data['book_table'] = pd.to_numeric(data['book_table'])
    
    # Label encode the categorical variables
    data['location'] = location_encoder.transform(data['location'])
    data['rest_type'] = RestType_encoder.transform(data['rest_type'])
    data['cuisines'] = cuisines_encoder.transform(data['cuisines'])
    
    return data

def train_extra_trees_regressor(X, Y, test_size, random_state, n_estimators):
    """
    Train an Extra Trees Regressor.

    Parameters:
    - X (pandas.DataFrame): The input features.
    - Y (pandas.Series): The target variable.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - ExtraTreesRegressor: The trained Extra Trees Regressor model.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    ET_model = ExtraTreesRegressor(n_estimators=n_estimators)
    ET_model.fit(x_train, y_train)
    
    y_predict = ET_model.predict(x_test)
    ext_score = round(r2_score(y_test, y_predict), 4)
    
    return ET_model, ext_score

def save_model(model, file_path):
    """
    Save a trained model to a file.

    Parameters:
    - model: The trained model object.
    - file_path (str): Path to save the model file.
    """
    pickle.dump(model, open(file_path, 'wb'))

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Zomato Restaurants Ratings Model Training')
    parser.add_argument('--data', type=str, default='zomato.csv', help='Path to the CSV data file')
    parser.add_argument('--location_encoder', type=str, default='location_encoder.pickle', help='Path to the location encoder pickle file')
    parser.add_argument('--RestType_encoder', type=str, default='RestType_encoder.pickle', help='Path to the RestType encoder pickle file')
    parser.add_argument('--cuisines_encoder', type=str, default='cuisines_encoder.pickle', help='Path to the cuisines encoder pickle file')
    parser.add_argument('--model', type=str, default='model.pkl', help='Path to save the trained model file')
    parser.add_argument('--test_size', type=float, default=0.3, help='Proportion of data for testing')
    parser.add_argument('--random_state', type=int, default=123, help='Random seed')
    parser.add_argument('--n_estimators', type=int, default=120, help='--model_n_estimators')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the label encoders
    try:
        with open(args.location_encoder, 'rb') as l_handle:
            location_encoder = pickle.load(l_handle)

        with open(args.RestType_encoder, 'rb') as l_handle:
            RestType_encoder = pickle.load(l_handle)

        with open(args.cuisines_encoder, 'rb') as l_handle:
            cuisines_encoder = pickle.load(l_handle)

        logging.info('Label encoders loaded successfully.')

        # Read the data
        data = read_data(args.data)
        
        # Preprocess the data
        preprocessed_data = preprocess_data(data, unwanted_columns, location_encoder, RestType_encoder, cuisines_encoder)

        # Splitting data
        X = preprocessed_data.drop(['rate'], axis=1)
        Y = preprocessed_data['rate']

        # Train the Extra Trees Regressor
        ET_model, ext_score = train_extra_trees_regressor(X, Y, test_size=args.test_size, random_state=args.random_state, n_estimators=args.n_estimators)

        # Save the model
        save_model(ET_model, args.model)

        # Save the model score and parameters to a text file
        score = 'r2 score: {}'.format(ext_score)
        parameters = f"n_estimators: {args.n_estimators}, test_size: {args.test_size}, random_state: {args.random_state}"
        with open('model_score_parameters.txt', 'w') as f:
            f.write(f"Model Score:\n{score}\n\nParameters:\n{parameters}")

        logging.info('Model training completed successfully.')
    except Exception as e:
        logging.error(f'Error occurred: {str(e)}', exc_info=True)

if __name__ == '__main__':
    main()