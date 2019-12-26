import os
import sys
import warnings
import numpy as np
import pandas as pd
from fastai.collab import *
from fastai.collab import CollabDataBunch
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support as score
warnings.filterwarnings('ignore')

class EmbeddedBiasModel:
    
    
        def __init__(self,review_data,model_name):
            self.review_data = review_data
            self.model_name = model_name
        
            
        def set_review_data(self,review_data):
            self.review_data = review_data
            
        def set_model_name(self,model_name):
            self.model_name = model_name
           
        def get_review_data(self):
            return self.review_data
        
        def get_model_name(self):
            return self.model_name

    
        def split_data(self):
            print("Splitting data into train and test ")
            print("\n")
            
            review_data = self.get_review_data()
            split_per = float(input("Enter the train - test split percentage : "))

            train, test = train_test_split(review_data, test_size=split_per, 
                                           stratify=review_data['user_id'])
            print("Completed ..")
            print("\n")

            return train,test
        

        """
        This function is used for building our model,where we create the 
        user and buisness embeddings each of size mentioned by the variable : factors
        ,we later find the 
        
        
        """
        def model_training(self,train,seed,factors):
            print(" Training EmbeddingDotBias Model , it might take a while ..")
    
            data = CollabDataBunch.from_df(train, seed= seed, user_name='user_id', item_name='business_id', rating_name='stars_review')
            learn1 = collab_learner(data, n_factors=factors, y_range=(0., 5.), wd=1e-1)
            
            print("Finding the learning rate.")
            print("\n")
            
            learn1.lr_find() 
            learn1.recorder.plot() 
            learn1.fit_one_cycle(40,3e-4)
            learn1.save(self.get_model_name())
             
            print(" Exporting the model.")
            print("\n")

            learn1.export()

            print(" Visalizing the results.")
            print("\n")

            learn1.show_results(rows=10)
            pd.set_option('display.max_columns', None)
    
            print("Completed ...")
            print("\n")

            return learn1
        
        def predict(self,test):
    
           print("Predicting ratings using the trained model ")
    
           learn2 = load_learner(".", test=CollabList.from_df(test, cat_names=['user_id', 'business_id'], path="."))
           display(learn2)
           preds, y = learn2.get_preds(ds_type=DatasetType.Test)
           preds1 =  preds.numpy()
           y = test["stars_review"].values
    
           print("Completed ...")
           print("\n")
    
           return preds1,y

  

        def evaluate(self,preds1,y):
        
            print("Evaluating the model")
        
            ms_error = mean_squared_error(y,preds1)
            
            print("MSE for the model : ", ms_error)
            
            preds2 = [ int(round(rating)) for rating in preds1]
            y2 = [ int(round(rating)) for rating in y]
            precision, recall, fscore, support = score(y2, preds2, average ="micro")
            
            print("\n")
            print('Precision: {}'.format(precision))
            print('Recall: {}'.format(recall))
            print('Fscore: {}'.format(fscore))
            print("\n")
            print("Completed ..")
    



if __name__ == "__main__":
    
    " Loading preprocessed, normalized data  - sampled from YelpData set."
    """
    1 - Normalized file name
    2 - train and test split 
    3 - model_name
    4 - seed value for collab_learner
    
    
    
    
    """
    
    review_data = pd.read_csv('task1_processed_data.csv')#sys.argv[1])
    model_name = "model1"
    seed = 35
    factors = 40
    
    model = EmbeddedBiasModel(review_data,"model_1")
    train,test = model.split_data()
    
    " Model 1 -  EmbeddingDotBias Model"

    " Training the model ,creating a collab_learner object."
    learn_obj = model.model_training(train,seed,factors)
   
    
    " Predicting the results on the test data."
    preds,y = model.predict(test)
    
    dict1 ={}
    count = 0
    for index, row in test.iterrows():
        if row["user_id"] == "ELcQDlf69kb-ihJfxZyL0A":
            dict1[row["business_id"]] = preds[count]
        count += 1
        
    print("\n")
    print("\n")
    print("Recommnded business for user id - ELcQDlf69kb-ihJfxZyL0A")
    count = 0
    for item in sorted(dict1.items(), key=lambda x: x[1] ,reverse =True):
        if count < 7:
            print("Business :"+ item[0] +"   " +"Likely Rating :"+ str(int(round(item[1]))))
            count += 1

    
    "Evaluate the model"
    model.evaluate(preds,y)
    
    
    
    

    





