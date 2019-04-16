from PyLyrics import *
import pandas


data = pandas.read_csv('final.csv')
colnames = list(data) 

df = pandas.read_csv("final.csv")
all_lyrics = []
count = 0

for x,y in zip(list(data['Artist']),list(data['Name'])):
    for i in range(1):
        all_lyrics.append(PyLyrics.getLyrics(x,y))
        count+=1
        # print(PyLyrics.getLyrics(x,y))
    break
    print("done")
    print(all_lyrics)

# print(PyLyrics.getLyrics('Taylor Swift','Blank Space')) #Print the lyrics directly
