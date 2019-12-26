
# Background

Are you sick of seeing primitive recommendation like these: 
![alt text][show]

[logo]: images/shoe.jpg

To explain in short, the idea behind this project is to build  a more effective recommendation system which goes beyond the status quo collaborative filtering based approach and utilize all possible data for recommendation. 


This project started out as an academic project for "Search" course taught by Prof. Xiaozhang Liu at Indiana university, Bloomington. The primary objective was to build a recommendation system using the yelp dataset, and recommend new businesses to users. Although the course is over, the project continuous in terms of improving the framework, code robustness as well as overall model performance. Readers are encourgaged to leave 

## The Yelp dataset
The Yelp dataset(https://www.yelp.com/dataset/challenge) is a collection of customer reviews, tips, checkins, item images as well as corresponding info on the users and businesses associated with the reviews. The businesses are although predominantly restuarants in the US, they also include salons, stores, spas and other business variety. This data can be used to recommend business/restuarant to a user on the basis of various criteria, for example, customers' history of visits and immediate geographical viscinity. 

## Primary Objective
Although, in theory, a general purpose recommendation system routine can be carried out on the entire dataset, it practically mnakes more sense to subset the data for a particular category of restuarants. We select a certain terms, and if any of the terms occure in the category tags for a business, we subset all such business and limit the data to all the reviews associated with them. Next we select a criteria for recommendation, just because a a user visits/checks in on a business doesn't mean that business is the best possible suggestion, and hence star rating is a better measure of affinity. 
Overall, we predict the star rating for user-business combination not already present in the avaibale data, and recommend the resturants in geogrphical proximity of the user sorted in decreasing order of predicted star rating. 

## Secondary Objective
Yelp is a useful service to the users, but can we utilize the prediction model of the primary objective for a more business oriented gain as well? Our prediction model utilizes not only the data of previous user-business visit history, but also user and business attributes. If we are able to have a quantitative model interpretation of how much each factor affects the prediction, we can utilize this value to guide busiiness startegy. Thus overall, we are not inly providing a service to for users to be able to enjoy the best services, but also a service to business owners who now know exactly what factors influence his nice/poort star rating and by how much.
We utilize algorithms like LIME and SHAP to implemnent this. For example, if we are to filter down the predictions for only one particular Pizza join, we will aggregate the feature impact values provided by LIME for this subset of data and the business attributes with positive




