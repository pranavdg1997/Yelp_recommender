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
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

class EmbeddingNNModel:
    
        def __init__(self,review_data,user_emb_size,buss_emb_size,model_name):
            self.review_data = review_data
            self.user_emb_size = user_emb_size
            self.buss_emb_size = buss_emb_size
            self.model_name = model_name
        
        def set_user_emb_size(self,user_emb_size):
            self.user_emb_size = user_emb_size
            
        def set_buss_emb_size(self,buss_emb_size):
            self.buss_emb_size = buss_emb_size
            
        def set_review_data(self,review_data):
            self.review_data = review_data
            
        def set_model_name(self,model_name):
            self.model_name = model_name
        
        def get_user_emb_size(self):
            return self.user_emb_size
        
        def get_buss_emb_size(self):
            return self.buss_emb_size
        
        def get_review_data(self):
            return self.review_data
        
        def get_model_name(self):
            return self.model_name

    
        def split_data(self):
            print("Splitting data into train and test ")
            print("\n")
            
            split_per = float(input("Enter the train - test split percentage : "))
            review_data = self.get_review_data()
            train, test = train_test_split(review_data, test_size=split_per, 
                                           stratify=review_data['user_id'])
            
            print("Completed ..")
            print("\n")

            return train,test
        

        def model_training(self,train,seed):
            
    
            
            user_emb = self.get_user_emb_size()
            buss_emb = self.get_buss_emb_size()
            
            data = CollabDataBunch.from_df(train, seed=seed, user_name='user_id', item_name='business_id', rating_name='stars_review')
            learn = collab_learner(data, use_nn=True, emb_szs={'user_id':user_emb ,'business_id':buss_emb}, layers=[256, 128], y_range=(0., 5.))
          
            
            learn.lr_find() 
            learn.recorder.plot() 
            learn.fit_one_cycle(40, 1e-2)
            learn.save(self.get_model_name())
                          
            
            learn.show_results(rows=10)
            pd.set_option('display.max_columns', None) 
            
            business_w = learn.model.embeds[1].weight[1:]
            buss_narray = business_w.cpu().data.numpy()
            mms_buss = MinMaxScaler()
            buss_narray = mms_buss.fit_transform(buss_narray)
            buss_ids = list(learn.data.train_ds.x.classes['business_id'][1:])

            self.business_df = pd.DataFrame(buss_narray,index=buss_ids)
            

            user_w = learn.model.embeds[0].weight[1:]
            user_narray = user_w.cpu().data.numpy()
            mms_user = MinMaxScaler()
            user_narray = mms_user.fit_transform(user_narray)
            user_ids = list(learn.data.train_ds.x.classes['user_id'][1:])

            self.user_df = pd.DataFrame(buss_narray,index=buss_ids)
            


            self.learn =learn            
            return
        
        
        def get_business_embeddings(self,buss_list):
            for buss in buss_list:
                if(buss not in self.business_df.index.values):
                    self.business_df.loc[buss,:]=np.average(self.business_df.values,axis=0)
            return(self.business_df.loc[buss_list,:].values)
            
        
        def get_user_embeddings(self,user_list):
            for user in user_list:
                if(user not in self.user_df.index.values):
                    self.user_df.loc[user,:]=np.average(self.user_df.values,axis=0)
            return(self.user_df.loc[user_list,:].values)
            
            
            
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
    model_name = "model2"
    seed = 35
    factors = 40
    
    model = EmbeddingNNModel(review_data,40,40,"model_2")
    train,test = model.split_data()
    
    " Model 2 -  EmbeddingNN Model"

    " Training the model ,creating a collab_learner object."
    learn_obj = model.model_training(train,seed)
   
    
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