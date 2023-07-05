# Restaurants Rating Prediction.

![image](https://user-images.githubusercontent.com/96771321/191105903-96e9a07f-2a31-402c-953b-e078e863da20.png)

> View Notebook @ https://github.com/Davidsonity/Restaurants-Rating-Prediction/blob/main/notebook.ipynb

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The objective of this project is to provide an accurate rating prediction for restaurants in Bangalore. By leveraging machine learning techniques, the project aims to help users make informed decisions when choosing a restaurant. The predictive model takes into account multiple factors, including location, cuisine, restaurant type, average cost, and user reviews, to generate reliable rating predictions.

## Dataset
The dataset used for training the model was sourced from Zomato. It provides comprehensive information about various restaurants in Bangalore, including their names, addresses, ratings, cuisines, average cost, user reviews, and more. The dataset was collected up until 15 March 2019. You can access the dataset on [Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants).

## Installation
To run this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/Davidsonity/Restaurants-Rating-Prediction.git`
2. Navigate to the project directory: `cd Restaurants-Rating-Prediction`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage
To use the Restaurants Rating Prediction application, follow these steps:

1. Ensure you have installed the required dependencies (see Installation section).
2. Run the application: `streamlit run app.py`
3. Access the application through your web browser at `http://localhost:8501`.
4. Provide the necessary restaurant details, such as location, cuisine, restaurant type, and average cost.
5. Optionally, you can input user reviews to further improve the prediction accuracy.
6. Click the "Predict" button to obtain the predicted rating for the restaurant.

## Deployment
The project is deployed and can be accessed online at [Restaurants Rating Prediction App](https://davidsonity-restaurants-rating-prediction-app-qsszth.streamlitapp.com/). The deployment allows users to access the application remotely without the need for local installation. It provides a user-friendly interface for predicting restaurant ratings.

## Project Structure
The project directory contains the following files and directories:

- `.idea`: Directory for IDE-specific configurations (e.g., PyCharm).
- `LICENSE`: The license file for the project.
- `Procfile`: A file used for specifying the application process type for deployment.
- `README.md`: This file, providing information about the project.
- `app.py`: The main application file that runs the Restaurants Rating Prediction app.
- `cuisines.json`: JSON file containing cuisine data used by the application.
- `location.json`: JSON file containing location data used by the application.
- `model.pkl`: Serialized machine learning model for rating prediction.
- `notebook.ipynb`: Jupyter Notebook containing the project's data exploration and model development.
- `requirements.txt`: Text file listing the required Python libraries and their versions.
- `rest_type.json`: JSON file containing restaurant type data used by the application.
- `setup.sh`: Shell script for environment setup.

## Contributing
Contributions to this project are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue on the GitHub repository. Your input will be highly appreciated in improving the functionality and accuracy of the rating prediction model.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code in accordance with the terms and conditions of the license.

