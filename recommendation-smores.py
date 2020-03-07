import os
import pandas as pd
import numpy as np
import time
import turicreate as tc
from boto import s3
from sklearn.model_selection import train_test_split
import sys
from flask import Flask, jsonify, request
import boto3

bucketName = 'smores-recommendation-engine'
sys.path.append("..")

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Smores Recommendation Engine</h1>'

@app.route('/search', methods=['GET'])
# Per each customer
def customer_recomendation():
    customer_id = (request.args['id'])

    if customer_id not in df_output.index:
        print('Customer not found.')
        return customer_id
    return df_output.loc[customer_id].to_json()

s3 = boto3.client(
    's3',
    # Hard coded strings as credentials, not recommended.
    aws_access_key_id=os.environ.get('AWS_KEY'),
    aws_secret_access_key= os.environ.get('AWS_SECRET'),
    region_name='us-west-2'
)

##Load Data
#List of 1000 users
s3.download_file(bucketName, 'recommend_1.csv', 'data/recommend_1.csv')
s3.download_file(bucketName, 'trx_data_copy.csv', 'data/trx_data.csv')
customers = pd.read_csv('data/recommend_1.csv')
#comsisting of user transactions
transactions = pd.read_csv('data/trx_data.csv')
customers.head()

# constant variables to define field names include:
user_id = 'customerId'
item_id = 'productId'
users_to_recommend = list(customers[user_id])
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset

def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy

def split_data(data):

    train, test = train_test_split(data, test_size=.2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data

def create_output(model, users_to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=users_to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id] \
        .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['customerId', 'recommendedProducts']].drop_duplicates() \
        .sort_values('customerId').set_index('customerId')
    if print_csv:
        df_output.to_csv('data/output/option1_recommendation.csv')
        s3.upload_file('data/output/option1_recommendation.csv', bucketName, 'output/option_recommendations.csv')
        print("An output file can be found in 'output' folder with name 'option1_recommendation.csv'")
    return df_output

# break down each list of items in the products column into rows and count the number of products bought by a user
transactions['products'] = transactions['products'].apply(lambda x: [(i) for i in x.split('|')])
transactions.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index()

# organize a given table into a dataframe with customerId, single productId, and purchase count
pd.melt(transactions.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index(),
             id_vars=['customerId'],
             value_name='products') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['customerId', 'products']) \
    .agg({'products': 'count'}) \
    .rename(columns={'products': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'products': 'productId'})


#This table will be an input for our modeling later In this case, our user is customerId, productId, and purchase_count
s=time.time()

data = pd.melt(transactions.set_index('customerId')['products'].apply(pd.Series).reset_index(),
             id_vars=['customerId'],
             value_name='products') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['customerId', 'products']) \
    .agg({'products': 'count'}) \
    .rename(columns={'products': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'products': 'productId'})
data['productId'] = data['productId']

# print(data.shape)
# data.head()


data_dummy = create_data_dummy(data)

#Now that we have three datasets with purchase dummy, we would like to split for modeling.
train_data_dummy, test_data_dummy = split_data(data_dummy)

final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy),
                                            user_id=user_id,
                                            item_id=item_id,
                                            target='purchase_dummy',
                                            similarity_type='cosine')

recom = final_model.recommend(users=users_to_recommend, k=n_rec)
df_rec = recom.to_dataframe()
df_rec.head()
df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id].transform(lambda x: '|'.join(x.astype(str)))
df_output = df_rec[['customerId', 'recommendedProducts']].drop_duplicates().sort_values('customerId').set_index('customerId')
df_output = create_output(final_model, users_to_recommend, n_rec, print_csv=True)
