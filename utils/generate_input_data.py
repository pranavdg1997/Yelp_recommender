import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


print("Reading from business.json")
business = pd.read_json("business.json",orient='columns',lines=True).dropna()
all_cats = list(business["categories"].unique())
print("All unique categories found = "+str(len(all_cats)))

term = "start"
terms = []
filtered_cats = set()

while(term!='stop'):
	term = input("Enter term you want in filtered output, enter 'stop' to stop: ")
	terms.append(term)
	term_cats = [cat for cat in all_cats if term in cat]
	print("Categories found for the term = "+str(len(term_cats)))
	filtered_cats = filtered_cats.union(set(term_cats))

filtered_cats = list(filtered_cats)
print("Categories found = "+str(len(filtered_cats)))
filtered_business = business.loc[business.categories.isin(filtered_cats),:]
print(str(filtered_business.shape[0])+" businesses found.")
del(business)

reviews = pd.read_json("review.json",lines=True,orient="columns")
reviews_by_business = reviews.loc[reviews.business_id.isin(filtered_business["business_id"].unique()),:]

print("Reviews filtered by business found = "+str(reviews_by_business.shape[0]))

filtered_reviews = pd.merge(reviews_by_business,filtered_business,on=["business_id"],how="outer",suffixes=["_review","_business"])
print(str(filtered_reviews.shape[0])+" reviews found in first merge")
del(reviews,filtered_business)

users = pd.read_json("user.json",lines=True,orient='columns')
filtered_users = users.loc[users["user_id"].isin(filtered_reviews.user_id.unique()),:]
user_filename = input("Enter filtered users filename: ")
filtered_users.to_csv(user_filename,index=False)
filtered_reviews = pd.merge(filtered_reviews,filtered_users,on=["user_id"],how="outer",suffixes=["_review","_user"])


del(users)

op_filename = input("Enter output filename: ")
review_dist = filtered_reviews.groupby("user_id")["review_id"].nunique().reset_index()
print("Reviews per user distribution: ")
print(review_dist["review_id"].describe())
k = int(input("Enter review per user count filter: "))
review_dist = review_dist.loc[review_dist["review_id"]>=k,:]
filtered_reviews = filtered_reviews.loc[filtered_reviews.user_id.isin(review_dist["user_id"].unique()),:]

print("Created filtered reviews dataset with "+str(filtered_reviews.shape[0])+" reviews.")
filtered_reviews.to_csv(op_filename,index=False)
del(filtered_reviews)

