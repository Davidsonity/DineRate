# Restaurants Rating Prediction

![image](https://user-images.githubusercontent.com/96771321/191105903-96e9a07f-2a31-402c-953b-e078e863da20.png)

> View Notebook @ https://github.com/Davidsonity/Restaurants-Rating-Prediction/blob/main/notebook.ipynb

This project is a machine learning application that predicts the ratings of restaurants in Bangalore. It utilizes a dataset obtained from Zomato and employs various features such as location, cuisine, restaurant type, average cost, and user reviews to train a predictive model.

The main objective of this project is to provide users with accurate rating predictions, aiding them in making informed decisions when selecting a restaurant. The model is deployed and accessible through a user-friendly interface on the deployment site.

## Features

- Predicts the rating of restaurants in Bangalore based on various factors.
- Utilizes a machine learning model trained on a Zomato dataset.
- Factors taken into account include location, cuisine, restaurant type, average cost, and user reviews.
- Provides users with accurate rating predictions to assist in restaurant selection.

## Dataset
The dataset used for training the model was sourced from Zomato. It provides comprehensive information about various restaurants in Bangalore, including their names, addresses, ratings, cuisines, average cost, user reviews, and more. The dataset was collected up until 15 March 2019. You can access the dataset on [Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants).

The dataset provides a rich set of information about restaurants, allowing the model to learn patterns and make accurate rating predictions.

## Deployment Site

The project is deployed and can be accessed through the following deployment site: [Restaurants Rating Prediction App](https://davidsonity-restaurants-rating-prediction-app-qsszth.streamlitapp.com/)

## Evaluation Metrics

The predictive model's performance is evaluated using the following metrics:

- R-squared (R2): The proportion of the variance in the target variable (restaurant ratings) that is predictable from the input variables. It indicates the goodness of fit of the model.

## Project Structure

The project directory contains the following files:

- `LICENSE`: The license file specifying the project's open-source license.
- `Procfile`: A file specifying the necessary commands to run the application on the deployment platform.
- `README.md`: The README file providing an overview of the project and instructions for setup and usage.
- `RestType_encoder.pickle`: Pickled object that encodes or transforms the "Restaurant Type" feature.
- `app.py`: The main Python script for running the application or script that utilizes the machine learning model.
- `cuisines_encoder.pickle`: Pickled object that encodes or transforms the "Cuisines" feature.
- `location_encoder.pickle`: Pickled object that encodes or transforms the "Location" feature.
- `model.pkl`: The trained machine learning model saved in a pickle file.
- `model_score_parameters.txt`: Text file containing the parameters or metrics related to the model's performance.
- `notebook.ipynb`: Jupyter Notebook containing the data preprocessing, model training, and evaluation steps.
- `requirements.txt`: Text file specifying the required Python packages and their versions.
- `setup.sh`: Shell script for setting up the required environment.
- `train_model.py`: Python script used for training the machine learning model.

This list should reflect the files present in your project directory based on the information you provided.
## Usage

To run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/Davidsonity/Restaurants-Rating-Prediction.git`
2. Navigate to the project directory: `cd Restaurants-Rating-Prediction`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the application: `python app.py`
5. Access the application in your web browser at `http://localhost:5000`

Make sure you have Python 3.x and pip installed on your system.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.



