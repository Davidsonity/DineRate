#IMPORT THE NECESSARY LIBRARIES
import streamlit as st
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


model_path = 'model.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

dic_online_order = {'Yes': 1, 'No': 0}

dic_book_table = {'Yes': 1, 'No': 0}

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

st.title("Zomato Restaurants Ratings")

st.image(
            "https://b.zmtcdn.com/data/pictures/7/19774377/7e38bb6b50847a67b2916293e74a8918_featured_v2.jpg" # Manually Adjust the width of the image as per requirement
        )

st.markdown("Check the rating of restaurants in Bangalore!!!")

order_online = st.selectbox("Do the restaurant offer online services?", dic_online_order.keys())
book_table = st.selectbox("Table Booking/Reservation Available?", dic_book_table.keys())
votes = st.slider("Number of votes", 0, 20000)
location = st.selectbox("Restaurant Location:", location_encoder.classes_)
rest_type = st.selectbox("Type of Restaurant", RestType_encoder.classes_)
cuisines = st.selectbox("Select cuisines", cuisines_encoder.classes_)
cost_for_2 = st.select_slider("Select price range for 2 customers",
                              options= np.linspace(40.0, 6000.0, 100))

if st.button("Submit"):
    answers = np.array([[dic_online_order[order_online],
           dic_book_table[book_table], 
           votes, 
           location_encoder.transform([location])[0],
           RestType_encoder.transform([rest_type])[0],
           cuisines_encoder.transform([cuisines])[0],
           cost_for_2]])
    st.write(answers)
    result = model.predict(answers)
    result_text = "The Restaurant's Rating is {} of 5.0.".format(round(result[0], 1))
    st.success(result_text)