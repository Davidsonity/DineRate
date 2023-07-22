#IMPORT THE NECESSARY LIBRARIES
import streamlit as st
import numpy as np
import json
import pickle
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Test the model
# Load the saved model
pipeline = pickle.load(open('model.pkl', 'rb'))

# Load the label encoder
with open('location_encoder.pickle', 'rb') as l_handle:
    location_encoder = pickle.load(l_handle)

# Load the label encoder
with open('RestType_encoder.pickle', 'rb') as l_handle:
    RestType_encoder = pickle.load(l_handle)
    
# Load the label encoder
with open('cuisines_encoder.pickle', 'rb') as l_handle:
    cuisines_encoder = pickle.load(l_handle)

                                          
####################################################################
#streamlit
##################################################################          

st.title("DineRate")

st.image(
            "https://b.zmtcdn.com/data/pictures/7/19774377/7e38bb6b50847a67b2916293e74a8918_featured_v2.jpg" # Manually Adjust the width of the image as per requirement
        )

st.markdown("Check the rating of restaurants in Bangalore!!!")

order_online = st.selectbox("Do the restaurant offer online services?", ['Yes', 'No'])
book_table = st.selectbox("Table Booking/Reservation Available?", ['Yes', 'No'])
votes = st.slider("Number of votes", 0, 20000)
location = st.selectbox("Restaurant Location:", location_encoder.classes_)
rest_type = st.selectbox("Type of Restaurant", RestType_encoder.classes_)
cuisines = st.selectbox("Select cuisines", cuisines_encoder.classes_)
cost_for_2 = st.select_slider("Select price range for 2 customers",
                              options= np.linspace(40.0, 6000.0, 100))



if st.button("Submit"):
    # Define the user inputs
    order_online = str(order_online)
    book_table = str(book_table)
    votes = int(votes)
    location = str(location)
    rest_type = str(rest_type)
    cuisines = str(cuisines)
    cost_for_2 = float(cost_for_2)

    # Create a dictionary with the user inputs
    data = {
        'online_order': [order_online],
        'book_table': [book_table],
        'votes': [votes],
        'location': [location],
        'rest_type': [rest_type],
        'cuisines': [cuisines],
        'cost_for_2': [cost_for_2]
    }

    # Create a DataFrame from the dictionary
    answers = pd.DataFrame(data)

    # Predict on the new data
    predictions = pipeline.predict(answers)
    
    predictions_text = "The Restaurant's Rating is {} of 5.0.".format(round(predictions[0], 1))
    st.success(predictions_text)
