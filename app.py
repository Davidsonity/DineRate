#IMPORT THE NECESSARY LIBRARIES
import streamlit as st
import numpy as np
import json

model_path = 'model.pkl'

import pickle
model = pickle.load(open(model_path,'rb'))

import warnings
warnings.filterwarnings('ignore')

dic_online_order = {'Yes': 1, 'No': 0}

dic_book_table = {'Yes': 1, 'No': 0}

# Opening JSON file
with open('location.json') as json_file:
    dic_location = json.load(json_file)

# Opening JSON file
with open('rest_type.json') as json_file:
    dic_rest_type = json.load(json_file)
    
# Opening JSON file
with open('cuisines.json') as json_file:
    dic_cuisines = json.load(json_file)

                                          
####################################################################
#streamlit
##################################################################          

st.title("Zomato Restaurants Ratings")

st.image(
            "https://b.zmtcdn.com/data/pictures/7/19774377/7e38bb6b50847a67b2916293e74a8918_featured_v2.jpg" # Manually Adjust the width of the image as per requirement
        )

st.markdown("Check the rating of restaurants in Bangalore!!!")

order_online = st.selectbox("Do the restaurant offer online services?", dic_online_order.keys())
book_table = st.selectbox("Table Booking/Reservation Available?", dic_book_table.keys())
votes = st.slider("Number of votes", 0, 20000)
location = st.selectbox("Restaurant Location:", dic_location.keys())
rest_type = st.selectbox("Type of Restaurant", dic_rest_type.keys())
cuisines = st.selectbox("Select cuisines", dic_cuisines.keys())
cost_for_2 = st.select_slider("Select price range for 2 customers",
                              options= np.linspace(40.0, 6000.0, 100))

answers = np.array([[
    dic_online_order[order_online], dic_book_table[book_table], votes,
    dic_location[location], dic_rest_type[rest_type], dic_cuisines[cuisines],
    cost_for_2
]])

if st.button("Submit"):
    result = model.predict(answers)
    result_text = "The Restaurant's Rating is {} of 5.0.".format(round(result[0], 1))
    st.success(result_text)