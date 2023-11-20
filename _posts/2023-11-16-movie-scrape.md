---
layout: post
title:  "Creating A Dataset Of Film Awards"
author: Carson Payne
description: Using web scraping and an API, here is how I created a dataset that contains information on the winners and nominees of film awards from the 21st century. 
image: /assets/images/film-awards.jpeg
---

Every year as the holiday season starts coming upon us, normal people start looking forward to Christmas, Hanukah, and the New Year but a select few other people are also looking forward to a different season- FILM AWARDS SEASON! Each year hundreds of films compete in dozens of different award shows for recognition of their craft. Often, winning a big award like an Oscar or Golden Globe propels filmmakers into new chapters in their careers, opening doors that were previously closed to them. No wonder big studios sometimes spend upwards of $40 million dollars just on campaigning their film to those voting on who wins the most coveted awards. 

With this in mind, I started wondering if it would be possible to actually predict who is most likely to win certain awards. After searching the internet for a dataset that would contain all the information I might need, I discovered there was nothing of the sort. So like any good data scientist, I got to work creating my own dataset with information on films from this century and how they performed in the film awards. 

## Part 1 - Scraping Awards Data
The first thing that I need to do is scrape data from each awards show. There are a couple of different sources on the internet to get this data. I could scrape directly from the award websites, Wikipedia, or iMDB. I chose to go with iMDB mostly because of the consistent format between awards shows. 

First, here are all the packages I used for this project that you can go ahead and load in.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

import re
import requests
import urllib.parse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer

from tqdm.notebook import tqdm
import time
from datetime import datetime, timedelta
```

Next, I went through and selected all the film awards I wanted to include in my dataset. You can see here that I included the event ID number from iMDB in a list. All the awards that are commented out in this list are because their formatting was a bit different than the rest. In the future, I can go through and make some alterations to my web scrapping code to work for each of the different awards.

```
guilds = ["ev0000003", #Oscars
          "ev0000123", #Baftas
          #"ev0000298", #Gothams
          "ev0000292", # Golden Globes
          #"ev0000133", # Critics Choice
          "ev0000349", #spirit
          #"ev0000147", # Cannes
          #"ev0000681", # Venice
          "ev0000631", # Sundance
          "ev0000598", #SAG
          "ev0000212", #DGA
          #"ev0000531", #PGA
          "ev0000618", #ADG
          "ev0000175", #CAS
          #"ev0000327", #MUAH
          "ev0000022", #ASC
          #"ev0000864", #VES
          #"ev0004323", #GMS
          "ev0000017", #Eddies
          "ev0000710" #WGA
    ]
```

Next, I went ahead and started to scrape the data from each of the awards and put them into the all_awards list. There is some commenting throughout the code chunk to explain what is going on here but essentially, all the awards from the 21st century are being added to this dataframe. 

```
all_awards = []

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

for l in range(len(guilds)):
    
    # Just going to scrape 21st century awards
    years_list = [str(year) for year in range(2000, 2024)]
    
    for k in range(len(years_list)):
        
        # Open the driver to the appropriate link
        url = "https://www.imdb.com/event/" + guilds[l] + "/" + years_list[k]
        driver.get(url)
        
        # Blank dataframe to fill in for each award
        ceremony = pd.DataFrame({'year': [''] * 1000, 'ceremony': [''] * 1000, 'award': [''] * 1000, 'id1': [''] * 1000, 'nom1': [''] * 1000, 'id2': [''] * 1000, 'nom2': [''] * 1000, 'id3': [''] * 1000, 'nom3': [''] * 1000, 'id4': [''] * 1000, 'nom4': [''] * 1000, 'nominated': [''] * 1000, 'winner': [''] * 1000})
        
        # Access the part of the page with the elements we want to scrape
        container = driver.find_elements(By.CLASS_NAME, 'event-widgets__award')[0]
        awards = container.find_elements(By.XPATH, ".//div[@class='event-widgets__award-category']")
        
        # Running row number to fill in the dataframe
        row = 0
        for j in range(len(awards)):
            
            # Select one of the awards from the page
            cur_award = awards[j]
            
            # Select the nominees from that award
            cur_noms = cur_award.find_elements(By.XPATH, ".//div[contains(@class, 'event-widgets__award-nomination')]")
            
            for i in range(len(cur_noms)):
                
                # Access certain easy to get data points
                ceremony["year"][i + row] = driver.find_element(By.CLASS_NAME, 'event-year-header__year').text
                ceremony["ceremony"][i + row] = driver.find_element(By.CLASS_NAME, 'event-widgets__award-name').text
                ceremony["award"][i + row] = cur_award.find_element(By.XPATH, ".//div[@class='event-widgets__award-category-name']").text
                
                # This next large chunk of code gets all of the links from each award section and copies them
                # So we can have each movie's ID later on
                link_elements = cur_noms[i].find_elements(By.XPATH, ".//a")
                if len(link_elements) >=1:
                    link = link_elements[0].get_attribute("href")
                    match = re.search(r'/([a-z]+)(\d+)/', link)
                    if match:
                        nom_prefix = match.group(1)
                        nom_id = match.group(2)
                        ceremony["id1"][i + row] = nom_prefix + nom_id
                if len(link_elements) >=2:
                    link = link_elements[1].get_attribute("href")
                    match = re.search(r'/([a-z]+)(\d+)/', link)
                    if match:
                        nom_prefix = match.group(1)
                        nom_id = match.group(2)
                        ceremony["id2"][i + row] = nom_prefix + nom_id
                if len(link_elements) >=3:
                    link = link_elements[2].get_attribute("href")
                    match = re.search(r'/([a-z]+)(\d+)/', link)
                    if match:
                        nom_prefix = match.group(1)
                        nom_id = match.group(2)
                        ceremony["id3"][i + row] = nom_prefix + nom_id
                if len(link_elements) >=4:
                    link = link_elements[3].get_attribute("href")
                    match = re.search(r'/([a-z]+)(\d+)/', link)
                    if match:
                        nom_prefix = match.group(1)
                        nom_id = match.group(2)
                        ceremony["id4"][i + row] = nom_prefix + nom_id
                nominees_elements = cur_noms[i].find_elements(By.XPATH, ".//a")
                if len(nominees_elements) >= 1:
                    ceremony["nom1"][i + row] = nominees_elements[0].text
                if len(nominees_elements) >= 2:
                    ceremony["nom2"][i + row] = nominees_elements[1].text
                if len(nominees_elements) >= 3:
                    ceremony["nom3"][i + row] = nominees_elements[2].text
                if len(nominees_elements) >= 4:
                    ceremony["nom4"][i + row] = nominees_elements[3].text
                
                # Marks that they were nominated and if they won
                ceremony["nominated"][i + row] = 1
                ceremony["winner"][i + row] = 1 if i == 0 else 0
            
            # Updates row number we are on in the dataframe
            row += len(cur_noms)
        
        # Removes blank rows from dataframe
        ceremony = ceremony.replace('', pd.NA).dropna(how='all').fillna('')
        
        # Adds this dataframe to a list of all the dataframes
        all_awards.append(ceremony)
```

After doing that web scrapping, I went ahead and concatenated all of the data frames and ended up with this huge list of information.

```
# Puts all of the dataframes together        
all_awards = pd.concat(all_awards, ignore_index=True)
```
<img width="866" alt="Screen Shot 2023-11-20 at 9 33 34 AM" src="https://github.com/carsonp4/carsonp4.github.io/assets/98862067/82157513-5f6a-492a-ac52-c63f941d87c9">

