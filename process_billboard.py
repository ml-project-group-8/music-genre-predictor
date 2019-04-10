
# coding: utf-8

# In[144]:


import pandas as pd
from ast import literal_eval


# In[149]:


df = pd.read_csv("data/billboard_2000_2018_spotify_lyrics.csv", encoding="latin1")


# In[167]:


def convert(x):
    def helper(y):
        a = y.split("'")
        return a[1] if len(a) > 1 else y
    return [helper(x[i]) for i in range(len(x))]

def to_list(x):
    """ 
        Converts a string representation of a list to a list
    """
    if x == "unknown":
        return ["unknown"]
    if type(x) == str:
        return literal_eval(x)
    return x

def simplify_genres(x, genre):
    pass


# In[168]:


df['genre'] = df.loc[:,'genre'].apply(to_list)


# In[179]:


pop_df = df[df.genre.apply(lambda x: 'country' not in x and 'pop' in x)]
cou_df = df[df.genre.apply(lambda x: 'pop' not in x and 'country' in x)]
both_df = df[df.genre.apply(lambda x: 'pop' in x and 'country' in x)]


# In[197]:


columns = ["date", "artist", "genre", "liveness", "acousticness", "instrumentalness", "time_signature", "duration_ms", "loudness", "lyrics"]
pop_df = pop_df[columns]
cou_df = cou_df[columns]
both_df = both_df[columns]


# In[198]:


pop_df.to_csv("data/billboards_processed.csv", sep=",", index=False)
cou_df.to_csv("data/billboards_processed.csv", sep=",", mode="a", index=False)

