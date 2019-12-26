import numpy as np
import pandas as pd
from text_preprocessing import *
import warnings
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error as mse
warnings.filterwarnings('ignore')

def convert_to_timedelta(df,date_col,td_col):
    DT = pd.to_datetime(df[date_col])
    min_date = DT.min()
    td_vals = DT.apply(lambda x:(x-min_date).days).values.astype(np.int)
    df[td_col] = td_vals - np.amin(td_vals)
    return(df)


def combine_reviews(review_texts):
    running_legacy_text = ''
    legacy_text = []
    for i in review_texts:
        legacy_text.append(running_legacy_text)
        running_legacy_text += " | " + i
    return(legacy_text)

input_filepath = input("Enter Input filepath: ")
total_data = pd.read_csv(input_filepath)


total_data = convert_to_timedelta(total_data,"date","timedelta")
total_data.sort_values(by=["user_id","business_id","timedelta"],inplace=True)
total_data = total_data.drop_duplicates(subset=["user_id","business_id"],keep="last")
total_data.sort_values(by=["user_id","timedelta"],inplace=True)

text_data = total_data[["user_id","review_id","timedelta","text","stars_review"]]
text_data["text"] = text_data["text"].apply(lambda x:text_cleaner(x))


text_data["legacy_text"] = text_data.groupby('user_id').text.transform(lambda x:combine_reviews(x)[:len(x)]).reset_index()["text"].values
tfv = TfidfVectorizer()
tfv.fit(text_data["text"].values)

X = tfv.transform(total_data_d["legacy_text"].values)
rs = Ridge(fit_intercept=True)
rs.fit(X,text_data['stars_review'].values)

n=1000
print("n="+str(n))
idx = np.argsort(np.abs(rs.coef_))[-n:]

Xc = X[:,idx].toarray()
names = [list(tfv.vocabulary_.keys())[id] for id in idx]

df_text = pd.DataFrame(data=Xc,columns=names)
output_filepath = input("Enter name of output file: ")
df_text.to_csv(output_filepath,index=False)







