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


## Part 2 - Cleaning the Scraped Data
Now that I have the different information on each of the film awards, I want to make this dataset more usable for machine learning. This involved cleaning this data a lot. Here is some of the things I did to make this dataset better.

The first thing I wanted to do was remove any sort of award from the dataset that did not pertain to the original question at hand. That included any sort of television awards, student/newcomer awards, or awards that were geo-specific. Here is the list of keywords I used to remove entries from my data:

```
# Removing different awards that aren't relevant to this analysis
film_awards = all_awards[~all_awards['award'].str.contains('Tele|Serie|Short|Commer|Show|Event|Breakthrough|Current|Music Video|British|Children|Daytime|Special Award|Student|Debut|First|Non-The|Under|Pilot|DVD|Reality|Musical|Talk|Variety Special', case=False, na=False)]
```

Next, I went through all of the remaining film awards and sorted them into shorter naming conventions. For example, I took every film that had anything to do with costume design and renamed it to costume_design. I could have been more explicit here for certain categories but for now, this will do. 

```
Adapted_Screenplay = ["Adapted Screenplay", "Best Adapted Screenplay", "Best Screenplay (Adapted)", "Best Screenplay - Adapted", "Best Screenplay Based on Material Previously Produced or Published", "Best Writing, Adapted Screenplay", "Best Writing, Screenplay Based on Material Previously Produced or Published"]
Animated = ["Animated Film", "Best Animated Feature", "Best Animated Feature Film", "Best Animated Feature Film of the Year", "Best Animated Featured Film", "Best Animated Film", "Best Motion Picture - Animated", "Outstanding Achievement in Sound Mixing for Motion Pictures - Animated"]
Art_Direction = ["Best Achievement in Art Direction", "Best Art Direction-Set Decoration"]
Casting = ["Best Casting", "Outstanding Performance by a Cast in a Motion Picture", "Outstanding Performance by a Cast in a Theatrical Motion Picture", "Outstanding Performance by the Cast of a Theatrical Motion Picture"]
Cinematography = ["Best Achievement in Cinematography", "Best Cinematography"]
Costume_Design = ["Best Achievement in Costume Design", "Best Costume Design"]
Directing = ["Best Achievement in Directing", "Best Director", "Best Director - Motion Picture"]
Documentary = ["Best Documentary", "Best Documentary Feature", "Best Documentary Film", "Best Documentary, Feature", "Best Documentary, Features", "Documentary", "Outstanding Achievement in Cinematography in Documentary Film", "Outstanding Achievement in Cinematography in Non-Fiction Filmmaking", "Outstanding Achievement in Sound Mixing for Motion Pictures - Documentary", "Outstanding Directorial Achievement in Documentary", "World Cinema - Documentary"]
Documentary_Screenplay = ["Best Documentary Screenplay", "Documentary Screenplay"]
Editing = ["Best Achievement in Film Editing", "Best Edited Feature Film - Comedy", "Best Edited Feature Film - Dramatic", "Best Editing", "Best Film Editing"]
Edited_Animated = ["Best Edited Animated Feature Film"]
Edited_Documentary = ["Best Edited Documentary", "Best Edited Documentary - Feature", "Best Edited Documentary - Theatrical", "Best Edited Documentary Film"]
Film = ["Best Feature", "Best Film", "Best Motion Picture - Drama", "Best Motion Picture of the Year", "Best Picture", "Contemporary Film", "Dramatic", "Fantasy Film", "Feature Film", "Outstanding Achievement in Cinematography in Feature Film", "Outstanding Achievement in Cinematography in Theatrical Feature Film", "Outstanding Achievement in Cinematography in Theatrical Releases", "Outstanding Achievement in Sound Mixing for Motion Pictures", "Outstanding Achievement in Sound Mixing for Motion Pictures - Live Action", "Outstanding Achievement in Sound Mixing for a Feature Film", "Outstanding Directorial Achievement in Feature Film", "Outstanding Directorial Achievement in Motion Pictures", "Outstanding Directorial Achievement in Theatrical Feature Film", "Period Film", "Period or Fantasy Film", "World Cinema - Dramatic"]
International = ["Best Foreign Film",  "Best International Feature Film", "Best International Film"]
Makeup_Hair = ["Best Achievement in Makeup", "Best Achievement in Makeup and Hairstyling", "Best Make Up & Hair", "Best Make Up/Hair", "Best Makeup", "Best Makeup and Hair"]
Non_English = ["Best Film Not in the English Language", "Best Film not in the English Language", "Best Foreign Language Film", "Best Foreign Language Film of the Year", "Best Motion Picture - Foreign Language", "Best Motion Picture - Non-English Language"]
Orignial_Screenplay = ["Best Original Screenplay", "Best Screenplay (Original)", "Best Screenplay - Original", "Best Screenplay Written Directly for the Screen", "Best Writing, Original Screenplay", "Best Writing, Screenplay Written Directly for the Screen", "Original Screenplay"]
Score = ["Best Achievement in Music Written for Motion Pictures (Original Score)", "Best Achievement in Music Written for Motion Pictures, Original Score", "Best Music, Original Score", "Best Original Music", "Best Original Score - Motion Picture", "Original Music", "Original Score"]
Screenplay = ["Best Screenplay", "Best Screenplay - Motion Picture"]
Song = ["Best Achievement in Music Written for Motion Pictures (Original Song)", "Best Achievement in Music Written for Motion Pictures, Original Song", "Best Music, Original Song", "Best Original Song - Motion Picture"]
Sound = ["Best Achievement in Sound Editing", "Best Achievement in Sound Mixing", "Best Sound", "Best Sound Editing", "Best Sound Mixing", "Outstanding Sound Mixing for Motion Pictures"]
Sound_Effects = ["Best Effects, Sound Effects Editing"]
Stunt = ["Outstanding Action Performance by a Stunt Ensemble in a Motion Picture", "Outstanding Performance by a Stunt Ensemble in a Motion Picture"]
Production_Design = ["Best Achievement in Production Design", "Best Production Design", "Best Production Design/Art Direction"]
Visual_Effects = ["Best Achievement in Special Visual Effects", "Best Achievement in Visual Effects", "Best Effects, Visual Effects", "Best Special Visual Effects", "Best Visual Effects"]

Lead = ["Best Lead Performance"]
Support = ["Best Supporting Performance"]
Lead_Actor = ["Best Actor in a Leading Role", "Best Leading Actor", "Best Male Lead", "Best Performance by an Actor in a Leading Role", "Best Performance by an Actor in a Motion Picture - Drama", "Outstanding Performance by a Male Actor in a Leading Role"]
Support_Actor = ["Best Actor in a Supporting Role", "Best Performance by an Actor in a Supporting Role", "Best Performance by an Actor in a Supporting Role in Any Motion Picture", "Best Performance by an Actor in a Supporting Role in a Motion Picture", "Best Supporting Actor", "Best Supporting Male", "Outstanding Performance by a Male Actor in a Supporting Role"]
Lead_Actress = ["Best Actress in a Leading Role", "Best Female Lead", "Best Leading Actress", "Best Performance by an Actress in a Leading Role", "Best Performance by an Actress in a Motion Picture - Drama", "Outstanding Performance by a Female Actor in a Leading Role"]
Support_Actress = ["Best Actress in a Supporting Role", "Best Performance by an Actress in a Supporting Role", "Best Performance by an Actress in a Supporting Role in Any Motion Picture", "Best Performance by an Actress in a Supporting Role in a Motion Picture", "Best Supporting Actress", "Best Supporting Female", "Outstanding Performance by a Female Actor in a Supporting Role"]
```

Then with these lists of award names, I could create a dictionary and then rename the awards in the dataframe.

```
# This creates a dictionary of the different awards from the work above

award_mapping = {
    "Adapted_Screenplay": Adapted_Screenplay,
    "Animated": Animated,
    "Art_Direction": Art_Direction,
    "Casting": Casting,
    "Cinematography": Cinematography,
    "Costume_Design": Costume_Design,
    "Directing": Directing,
    "Documentary": Documentary,
    "Documentary_Screenplay": Documentary_Screenplay,
    "Editing": Editing,
    "Edited_Animated": Edited_Animated,
    "Edited_Documentary": Edited_Documentary,
    "Film": Film,
    "International": International,
    "Makeup_Hair": Makeup_Hair,
    "Non_English": Non_English,
    "Orignial_Screenplay": Orignial_Screenplay,
    "Score": Score,
    "Screenplay": Screenplay,
    "Song": Song,
    "Sound": Sound,
    "Sound_Effects": Sound_Effects,
    "Stunt": Stunt,
    "Production_Design": Production_Design,
    "Visual_Effects": Visual_Effects,
    "Lead": Lead,
    "Support": Support,
    "Lead_Actor": Lead_Actor,
    "Support_Actor": Support_Actor,
    "Lead_Actress": Lead_Actress,
    "Support_Actress": Support_Actress
}

for new_value, old_values in award_mapping.items():
    film_awards['award'] = film_awards['award'].replace(old_values, new_value)
```

The next thing I needed to do was make sure that every entry in the list had an iMDB film ID. I scraped the first 3 id's from each nomination but a few (looking at you Producers Guild of America) awards listed a lot of people before the film. So I manually went ahead and found the missing film ID for the few entries.

```
film_awards.at[8300, 'id2'] = "tt0243017"
film_awards.at[8301, 'id2'] = "tt0248845"
film_awards.at[8302, 'id2'] = "tt0245501"
film_awards.at[8303, 'id2'] = "tt0242587"
film_awards.at[8351, 'id2'] = "tt0282864"
film_awards.at[8353, 'id2'] = "tt0274622"
film_awards.at[12774, 'id2'] = "tt0169547"
film_awards = film_awards.drop(12810)
film_awards.at[12819, 'id2'] = "tt0190332"
film_awards.at[12834, 'id2'] = "tt0268978"
film_awards.at[12912, 'id2'] = "tt0299658"
film_awards.at[12958, 'id2'] = "tt0167260"
film_awards.at[13006, 'id2'] = "tt0405159"
film_awards.at[13050, 'id2'] = "tt0388795"
film_awards.at[13379, 'id2'] = "tt1024648"
film_awards.at[13469, 'id2'] = "tt1454468"
film_awards.at[13519, 'id2'] = "tt2562232"
film_awards.at[13559, 'id2'] = "tt1663202"
film_awards.at[13563, 'id2'] = "tt1895587"
film_awards.at[13679, 'id2'] = "tt5580390"
```

Then I went ahead and grabbed the film ID from each row and made that a new column. I then selected the columns I needed from the scrapped data, cleaned up the year variable, and renamed the film awards to be a bit more reader-friendly. 

```
# Making a column of all the film id's
film_awards['tt_values'] = film_awards.apply(lambda row: list(set([value for value in row if isinstance(value, str) and value.startswith('tt')])), axis=1)

# Selecting the film_id from each movie
film_awards['film_id'] = film_awards['tt_values'].apply(lambda x: x[0] if x else None)

# Selecting all the columns we want to use now
clean_df = film_awards[["year", "ceremony","award", "nominated", "winner", "film_id"]]

# Cleaning the year varuable to just be the year digits
clean_df['year'] = clean_df['year'].str.extract('(\d+)', expand=False)

# Renamng the award names to be more reader friendly
clean_df['ceremony'] = clean_df['ceremony'].replace('BAFTA Film Award', 'BAFTA')
clean_df['ceremony'] = clean_df['ceremony'].replace('Golden Globe', 'GG')
clean_df['ceremony'] = clean_df['ceremony'].replace('Independent Spirit Award', 'Spirit')
clean_df['ceremony'] = clean_df['ceremony'].replace('Grand Jury Prize', 'Sundance')
clean_df['ceremony'] = clean_df['ceremony'].replace('Actor', 'SAG')
clean_df['ceremony'] = clean_df['ceremony'].replace('DGA Award', 'DGA')
clean_df['ceremony'] = clean_df['ceremony'].replace('Excellence in Production Design Award', 'ADG')
clean_df['ceremony'] = clean_df['ceremony'].replace('C.A.S. Award', 'CAS')
clean_df['ceremony'] = clean_df['ceremony'].replace('ASC Award', 'ASC')
clean_df['ceremony'] = clean_df['ceremony'].replace('WGA Award (Screen)', 'WGA')
```

