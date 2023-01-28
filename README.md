# Restaurant Ratings Prediction.

![image](https://user-images.githubusercontent.com/96771321/191105903-96e9a07f-2a31-402c-953b-e078e863da20.png)


> Deployment Site @ https://davidsonity-restaurants-rating-prediction-app-qsszth.streamlitapp.com/
> View Notebook @ https://github.com/Davidsonity/Restaurants-Rating-Prediction/blob/main/notebook.ipynb

### INTRODUCTION
#### Objectives
The objective of this project is to build machine learning model to predict the rating of restaurants in Bangalore.

#### About Dataset
The data is accurate to that available on the zomato website until 15 March 2019.
The data was scraped from Zomato in two phase. After going through the structure of the website I found that for each neighborhood there are 6-7 category of restaurants viz. Buffet, Cafes, Delivery, Desserts, Dine-out, Drinks & nightlife, Pubs and bars.

Phase I,

In Phase I of extraction only the URL, name and address of the restaurant were extracted which were visible on the front page. The URl's for each of the restaurants on the zomato were recorded in the csv file so that later the data can be extracted individually for each restaurant. This made the extraction process easier and reduced the extra load on my machine. The data for each neighborhood and each category can be found here

Phase II,

In Phase II the recorded data for each restaurant and each category was read and data for each restaurant was scraped individually. 15 variables were scraped in this phase. For each of the neighborhood and for each category their onlineorder, booktable, rate, votes, phone, location, resttype, dishliked, cuisines, approxcost(for two people), reviewslist, menu_item was extracted.

Data Source: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants

Columns Description: The following are the description of each columns in the dataset:
- url: zomato url for the restaurants
- address: complete location of the restaurant
- name: name of the restaurant
- online_order: whether restaurant accepts online order
- book_table: whether restaurant provides option for booking table
- rate: restaurants rating on zomato website
- votes: number of individual who voted for restaurants
- phone: contact details of the restaurant
- localtion: area where restaurant is situated
- rest_type: Type of restaurants (Categorical value)
- dish_liked: what are all dishes of the restaurant that people liked
- cuisines: cuisines offered by the restaurant
- approx_cost(for two people): average cost for two people
- review_list: reviews of the restaurant on zomato website
- menu_item: menu items available in the restuarant
- listed_in(type): type of the restaurant
- listed_in(city): locality of the restaurant position


### WEBSITE
https://davidsonity-restaurants-rating-prediction-app-qsszth.streamlitapp.com/
