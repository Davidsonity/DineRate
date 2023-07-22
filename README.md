# DineRate
> Restaurants Rating Prediction

![image](https://user-images.githubusercontent.com/96771321/191105903-96e9a07f-2a31-402c-953b-e078e863da20.png)

> View Notebook @ https://github.com/Davidsonity/DineRate/blob/main/notebook.ipynb

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

The project is deployed and can be accessed through the following deployment site: [DineRate App](https://dinerate.streamlit.app/)

## Evaluation Metrics

The predictive model's performance is evaluated using the following metrics:

- R-squared (R2): The proportion of the variance in the target variable (restaurant ratings) that is predictable from the input variables. It indicates the goodness of fit of the model.

## Project Structure

The project directory contains the following files:

- `LICENSE`: This file contains the license information for your project, specifying how others can use and distribute your code.
- `Procfile`: This file declares the commands to start and run your application, typically used in deploying applications to platforms like Heroku.
- `README.md`: This file provides a description, instructions, and other relevant information about your project. It is often written in Markdown format.
- `RestType_encoder.pickle`, `cuisines_encoder.pickle`, `location_encoder.pickle`: These pickle files contain pre-trained encoders or label encodings for specific features in your machine learning model. They are used to encode categorical variables.
- `app.py`: This file contains the code for your application or API. It defines routes, handles requests, and may perform tasks such as loading the model and making predictions.
- `model.pkl`: This file is a pickled version of your trained machine learning model. It is a serialized form of your TensorFlow or scikit-learn model that can be loaded and used for inference.
- `notebook.ipynb`: This Jupyter Notebook file contains code, documentation, and visualizations related to your project. It is typically used for exploratory data analysis (EDA), data preprocessing, or model development.
- `final_model.ipynb`: This Jupyter Notebook file contains code, documentation, and visualizations related to your final model development. It likely includes the refined model architecture, hyperparameter tuning, and model evaluation on test data.
- `requirements.txt`: This file lists the required Python packages and their specific versions needed to run your project. It ensures that the correct dependencies are installed.
- `setup.sh`: This shell script is used for setting up your project environment. It may include commands to install dependencies, configure environment variables, or perform other setup tasks.
- `train_model.py`: This file contains the code for training your machine learning model. It includes data loading, preprocessing, model creation, training, and model evaluation.

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



