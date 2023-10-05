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
