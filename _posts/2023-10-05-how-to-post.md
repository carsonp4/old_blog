---
layout: post
title:  "How to build a machine learning model to give personalized movie recommendations"
author: Carson Payne
description: Learn how to download your IMDb data and build a machine-learning model to give personalized movie recommendations.    
image: "/assets/images/ferris.webp"
---

For over a year now, every time I watch a movie I hop on to IMDb to give the movie a score from 1-10 on how much I enjoy it. I have now rated over 300 titles and being a data scientist, want to know if I can build myself a database of movie recommendations based on the films I have liked historically. So, if you have rated some films on IMDb before, here I present to you code that you can easily replicate to produce your own recommendations.

## Step 1 - Gathering Data
First, we need to obtain both a training dataset and a testing dataset. For the training set we will use your own movie recommendations. To obtain this, go to your IMDb account, navigate to the ['Your Ratings' page](https://www.imdb.com/list/ratings/?ref_=helpms_ih_tm_votesfaqs), click on the three dots, and click 'Export'. If you don't have any films rated on IMDb and you want to replicate this project, you can use this user's data [here](https://github.com/carsonp4/carsonp4.github.io/blob/main/assets/Keith%20Ratings.csv).

For the testing set we just need a list of movies that we want to make movie recommendations from. Some lists that I found which can be downloaded from IMDb are:

[Top 1000 Movies Ever Made](https://www.imdb.com/list/ls048276758/)

[Top 1000 Highest Grossing Films Ever](https://www.imdb.com/list/ls098063263/)

[Every Disney Movie Every Made](https://www.imdb.com/list/ls026785255/)

[Every Movie Every Nominated For An Oscar](https://www.imdb.com/list/ls055903720/)

You can also download [this file](https://github.com/carsonp4/carsonp4.github.io/blob/main/assets/big_list.csv) which is a combination of all four of these lists and is the testing dataset I will use for the rest of this tutorial. 



## Step 2 - Data Manipulation

Once the data has been downloaded, go ahead and open a python file and load the following packages as well as the data.

```
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from datetime import date

ratings = pd.read_csv("ratings.csv") # training data
biglist = pd.read_csv("big_list.csv") #testing data
```

The first thing we want to do is make sure to remove any movie in the testing dataset that exists in the training data so we don't recommend movies that we have already seen.

```
biglist = biglist[~biglist['Const'].isin(ratings['Const'])] # Removing watched movies from the testing set
```

Next we'll make sure to select only movies in the list, since there could be tv shows in the set, as well as select the columns we want to use as training and response variables. I created a join table that we'll join with the predicitions later in the project to see which movies the model recommends.  I also renamed the columns for a bit of help later on.

```
ratings = ratings[ratings['Title Type'] == 'movie'].iloc[:, [1, 6, 7, 9, 10, 11, 12]] # Filtering for only movies and selecting columns

biglist = biglist[biglist['Title Type'] == 'movie'] # Filtering for only movies
join = biglist.iloc[:, [5, 8, 9, 11, 12, 13, 14]] # Saving movie name data for later
biglist = biglist.iloc[:,[4,8,9,11,12,13,14]] # Selecting columns

ratings.columns = ["Rating", "IMDB", "Runtime", "Genres", "NumVotes", "Release", "Directors"] # Renaming Columns
biglist.columns = ["Rating", "IMDB", "Runtime", "Genres", "NumVotes", "Release", "Directors"] # Renaming Columns
```

Now we are going to combine the testing and training data for some feature engenieering. For example, first we are going to change the release data column to days sinced the movie was released. 

```
combined = pd.concat([ratings, biglist], axis=0, ignore_index=True) # Combining training and testing data for feature engenieering

combined["Days"] = (pd.to_datetime(date.today()) - pd.to_datetime(combined['Release'])).dt.days # Creating days since released column
combined.drop(columns=["Release"], inplace=True) # Removing release date column
```
