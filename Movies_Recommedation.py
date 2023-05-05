#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os 
import warnings

import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
init_notebook_mode(connected=True)


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


cwd = os.getcwd()


df=pd.read_csv("C:/Users/sonam/OneDrive - San Diego State University (SDSU.EDU)/ML/movies.csv")
df.head()

df['YEAR'].unique()

df['YEAR'] = df.YEAR.apply(lambda x: str(str(x).split("(")[-1]).split(")")[0])
years= df.YEAR

## years that end in "-" means that the show is still going,so its better to take this account in order to update it later on to current year.
with_hypen=[]
for i in range(len(years)):
    if years[i][-1]==" ":
        with_hypen.append(1)
    else:
        with_hypen.append(0)
df['hypen_year']=with_hypen

def convert(list):
      
    # Converting integer list to string list
    s = [str(i) for i in list]
      
    # Join list items using join()
    res = "".join(s)
      
    return(res)

year=[]
year_1=[]
year_2=[]
for i in range(len(df)):
    a = re.findall("\w\d",df.YEAR[i])
    year.append(convert(a))
    year_1.append(year[i][:4])
    year_2.append(year[i][-4:])

df['YEAR']=year
df['beg_year']=year_1
df['end_year']=year_2

## Converting values to numeric values
df['beg_year']=pd.to_numeric(df['beg_year'])
df['end_year']=pd.to_numeric(df['end_year'])

hypen=[]
for i in range(len(df)):
    if df['hypen_year'][i]==1:
        hypen.append(2022)
    else:
        hypen.append(df['end_year'][i])
df['end_year']=hypen

df['Num_years']=df['end_year']-df['beg_year']
df.beg_year.fillna(0,inplace=True)
df.end_year.fillna(0,inplace=True)
df.Num_years.fillna(0,inplace=True)

df['VOTES']=df['VOTES'].str.replace(",","")
df['VOTES']=pd.to_numeric(df['VOTES'])
df['VOTES']=df['VOTES'].fillna(df.VOTES.median())

df['RunTime']=df['RunTime'].fillna(df.RunTime.median())

df['RATING']=df['RATING'].fillna(df.RATING.median())

df['Gross'].fillna("0",inplace=True)

df['GENRE'] = df['GENRE'].apply(lambda x: str(x).split("\n")[-1]) ## to remove "\n" and fetch the value of the genre's

df['GENRE'] = df['GENRE'].str.replace(' ', '') ## There are multiple unncessary spaces in genre 

df.head()


## Counting the number of movies in which each genre is featured

from itertools import combinations
from collections import Counter

count=Counter()

for row in df['GENRE']:
    row_split=row.split(",")
    count.update(Counter(combinations(row_split,1)))
    
count.most_common(10)

genre=[]

for i,key in enumerate(count):
    genre.append(str(key).split("(")[-1].split("'")[1]) ## getting rid of unncessary parts of the string

## creating columns for each genre and updating the respecting values

df[genre]=0

for i in range(len(df)):
    for j in range(len(genre)):
        df_temp=pd.DataFrame([df.GENRE[i]],columns=["genre"]) ## creating a temperory dataframe for using str.contains function on each row of genre.
        if df_temp['genre'].str.contains(genre[j]).sum()==1:
            df[genre[j]][i]=1
        else:
            pass
        
df_new =df.copy() 

df['director']=df['STARS'].apply(lambda x: str(x).split("Star")[0])
df['actors'] = df['STARS'].apply(lambda x: str(x).split("Star")[-1])

df['actors']=df['actors'].str.replace("\n","").str.replace("s:","")
df['director']= df['director'].apply(lambda x: x.split(":")[-1]).str.replace("\n","").str.replace("|","")

famous_actors = df['actors'].str.lower().str.replace(" ","")

from itertools import combinations
from collections import Counter

count=Counter()
for row in famous_actors:
    row_split=row.split(",")
    count.update(Counter(combinations(row_split,1)))
    
count.most_common(10)

df_cleaned = df.drop(["STARS",'YEAR', 'hypen_year', 'end_year'],axis=1)
df_cleaned.drop_duplicates("MOVIES",inplace=True)

df_cleaned.head()



# In[13]:


df['Gross']=df['Gross'].str.replace("$","").str.replace("M","")
df['New_Gross'] = pd.to_numeric(df['Gross'])
df['New_Gross']=df['New_Gross']*1000000
gross_amount = df['Gross'].sort_values(ascending=False).values[:10]
gross_amount = [float(i)*1000000 for i in gross_amount]
idx = df['Gross'].sort_values(ascending=False).index[:10]
movies=df.iloc[idx]['MOVIES']

plt.figure(figsize=(10,5))
sns.barplot(x=movies,y=gross_amount,palette="winter")
plt.xlabel("Grossing_amount")
plt.ylabel("Movies")
plt.xticks(rotation=90)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}'.format(x) for x in current_values])
plt.title(label="TOP GROSSING MOVIES")


# In[3]:


movies = df_cleaned.groupby("MOVIES")['VOTES'].sum().sort_values(ascending=False).reset_index(name="VOTES")[0:10]
plt.figure(figsize=(12,7))
font = {'color':  'navy',
        'weight': 'normal',
        'size': 20,
        }
sns.barplot(y='MOVIES',x='VOTES',data=movies,palette="icefire")
plt.ylabel("Movies",fontdict=font)
plt.xlabel("Number of Votes",fontdict=font)
plt.yticks(rotation=0)
current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:.0f}'.format(x) for x in current_values])
plt.title(label="MOST VOTED MOVIES",fontdict=font)


# In[4]:


from itertools import combinations
from collections import Counter

count=Counter()

for row in df_cleaned['GENRE']:
    row_split=row.split(",")
    count.update(Counter(combinations(row_split,1)))
genre_counter = count
genre_count = dict(count.most_common(10))


# In[5]:


genres = list(genre_count.keys())
genres = [str(a).replace(",","").replace(")","").replace("(","").replace("'","") for a in genres]

count_of_genres = list(genre_count.values())


plt.figure(figsize=(10,5))
font = {'color':  'navy',
        'weight': 'normal',
        'size': 20,
        }

sns.barplot(x=genres,y=count_of_genres,palette="dark")
plt.xlabel("Genre's",
           fontdict=font)

plt.ylabel("Popualarity"
           ,fontdict=font)

plt.xticks(rotation=90)

current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}'.format(x) for x in current_values])

plt.title(label="MOST POPULAR GENRE",fontdict=font)


# In[6]:


genre_raw = df_cleaned['GENRE'].dropna().to_list()
genre_list = list()

for genres in genre_raw:
    genres = genres.split(",")
    for g in genres:
        genre_list.append(g)
        
genre_df = pd.DataFrame.from_dict(Counter(genre_list), orient = 'index').rename(columns = {0:'Count'})
genre_df.head()


# In[7]:


fig = px.pie(data_frame = genre_df,
             values = 'Count',
             names = genre_df.index,
             color_discrete_sequence = px.colors.qualitative.Safe)

fig.update_traces(textposition = 'inside',
                  textinfo = 'label+percent',
                  pull = [0.05] * len(genre_df.index.to_list()))

fig.update_layout(title = {'text':'Genre Distribution'},
                  legend_title = 'Genre',
                  uniformtext_minsize=13,
                  uniformtext_mode='hide',
                  font = dict(
                      family = "Courier New, monospace",
                      size = 18,
                      color = 'black'
                  ))


fig.show()


# In[10]:


# Features  using GENRE, RATING??, ONE-LINE, RunTime??, Director, Stars

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

features = ['GENRE','ONE-LINE','director','actors']

# Filling in missing values with Blank String
for feature in features:
    df_cleaned[feature] = df_cleaned[feature].fillna("")

df_cleaned['combined_features'] = df_cleaned['GENRE'] + " " + df_cleaned['ONE-LINE'] + " " + df_cleaned['director'] + " " + df_cleaned['actors'] 
cv = CountVectorizer()
count_matrix = cv.fit_transform(df_cleaned['combined_features'])
cosine_sim = cosine_similarity(count_matrix)


# In[11]:


def movie_recommendation(mov,sim_num = 5):

    user_choice = mov
    
    try:
        ref_index = df_cleaned[df_cleaned['MOVIES'].str.contains(user_choice, case = False)].index[0]

        similar_movies = list(enumerate(cosine_sim[ref_index]))

        sorted_simmilar_movies = sorted(similar_movies, key = lambda x: x[1], reverse = True)[1:]

        print('\nRecomended Movies for [{}]'.format(user_choice))
        print('-'*(24 + len(user_choice)))

        for i, element in enumerate(sorted_simmilar_movies):
            similar_movie_id = element[0]
            similar_movie_title = df_cleaned['MOVIES'].iloc[similar_movie_id]
            s_score = element[1]
            print('{:40} -> {:.3f}'.format(similar_movie_title, s_score))

            if i > sim_num:
                break
    except IndexError:
        print("\n[{}] is not in our database!".format(user_choice))
        print("We couldn't recommend anyting...Sorry...")


# In[12]:


movie_recommendation("spider",sim_num=10)


# In[ ]:





# In[ ]:




