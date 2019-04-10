# requests
import re
import urllib.request
import requests
import csv
from collections import defaultdict
from bs4 import BeautifulSoup
import pandas


def get_lyrics(artist,song_title):
    """
    uses azlyrics to extract lyrics
    """
    artist = artist.lower()
    song_title = song_title.lower()
    # remove all except alphanumeric characters from artist and song_title
    artist = re.sub('[^A-Za-z0-9]+', "", artist)
    song_title = re.sub('[^A-Za-z0-9]+', "", song_title)
    if artist.startswith("the"):    # remove starting 'the' from artist e.g. the who -> who
        artist = artist[3:]
    url = "http://azlyrics.com/lyrics/"+artist+"/"+song_title+".html"
    
    try:
        content = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(content, 'html.parser')
        lyrics = str(soup)
        # lyrics lies between up_partition and down_partition
        up_partition = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->'
        down_partition = '<!-- MxM banner -->'
        lyrics = lyrics.split(up_partition)[1]
        lyrics = lyrics.split(down_partition)[0]
        lyrics = lyrics.replace('<br>','').replace('</br>','').replace('</div>','').strip()
        return lyrics
    except Exception as e:
        return "Exception occurred \n" +str(e)

colnames = ['Genre','ID','popularity','is_exp','name','danceability','energy','key','loudness',
'mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature']

data = pandas.read_csv('final.csv')
colnames = list(data) #['Unnamed: 0', 'Genre', 'ID', 'popularity', 'is_exp', 'name', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
# for x,y in (data[['artist', 'name']]):
# print(data.iloc[0:,0:2]) 
with open('final.csv','r') as csvinput:
    with open('lyrics.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
            row.append(get_lyrics(x,y))
            all.append(row)

        for row in reader:
            row.append(row[0])
            all.append(row)

        writer.writerows(all)
