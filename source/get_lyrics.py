import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
from config import genius_id, genius_secret, genius_token
import time
import base64
from config import client_id, client_secret

def get_token():
    b64val = str.encode("{}:{}".format(client_id,client_secret))
    base64encoded = base64.standard_b64encode(b64val).decode("utf-8")

    auth_body = {
        "grant_type": "client_credentials"
    }
    auth_head = {
        "Authorization": "Basic {}".format(base64encoded)
    }

    auth_url = "https://accounts.spotify.com/api/token"

    timeBefore = time.time()
    response = requests.post(auth_url, headers=auth_head, data=auth_body)

    code = response.status_code
    data = response.json()

    if "access_token" not in data:
        print("Could not login. Exiting")
        return None
    token = data["access_token"]
    return token

def parse_info(art, tit):
    art = art.strip()
    if "--" in tit:
        tit = tit[:tit.index("--")]
    tit=tit.strip()
    artist = ""
    title  = ""

    for char in art:
        if char == " " or char.isalpha() or char.isnumeric():
            artist += char

    for char in tit:
        if char == " " or char.isalpha() or char.isnumeric():
            title += char

    artist = re.sub("[\(\[].*?[\)\]]", "", artist).strip()
    title  = re.sub("[\(\[].*?[\)\]]", "", title).strip()
    return artist, title

base_url  ="https://api.genius.com/search"
params = {
    "q": None
}
headers = {
    "Authorization":"Bearer {}".format(genius_token)
}

def grab_lyrics(query):
    #a,t = parse_info(artist,title)
    #query = a + t
    params["q"] = query

    response= requests.get(base_url, params=params,headers=headers)
    if response.status_code != 200:
        response= requests.get(base_url, params=params,headers=headers)
        if response.status_code != 200:
            return False, "Couldn't access Genius", ""

    json = response.json()
    results = json.get("response").get("hits")
    if type(results) != list or len(results) == 0:
        return False, "Results in incorrect format", ""

    results = results[0]
    if type(results) != dict:
        return False, "Wrong results type",""

    lyric_path = results.get("result").get("path")

    if lyric_path == None:
        return False, "Could not get lyrics url", ""

    lyric_url = "https://genius.com"+lyric_path
    response = requests.get(lyric_url)

    if response.status_code != 200:
        return False, "Error grabbing lyrics", ""

    soup = bs(response.text, "html.parser")
    lyrics = soup.find("p")


    if lyrics == None:
        return True, "Success", ""

    lyrics = lyrics.get_text()
    lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics)
    lyrics = lyrics.replace("\n"," ")
    lyrics = lyrics.replace("\t"," ")
    lyrics = lyrics.strip()
    return True, "Success", lyrics

def grab_features(query):
    data = {
        "q": query,
        "type": "track",
        "limit": 1
    }

    token   = get_token()

    headers = {
        "Authorization": "Bearer {}".format(token),
    }

    response = requests.get("https://api.spotify.com/v1/search",headers=headers,
                            params=data)
    items = response.json().get("tracks").get("items")
    if len(items) == 0:
        return False, "No search results", ""

    ident = items[0].get("id")
    tr_url = "https://api.spotify.com/v1/tracks/{}".format(ident)

    tr_response = requests.get(tr_url, headers=headers)
    track_data = []
    try:
        t = tr_response.json()
        # ID, Name, Popularity, Explicit, Artist
        track_data += [
            t["id"],t["name"],t["popularity"],t["explicit"],
            t["artists"][0]["name"]
        ]
    except:
        return False, "Bad track data",""
    track_url    = "https://api.spotify.com/v1/audio-features/{}".format(track_data[0])

    prev_names = [
        "Id","Name","Popularity","Is_Exp","Artist"
    ]
    feat_names = [
        "danceability","energy","key","loudness","mode","speechiness",
        "acousticness","instrumentalness","liveness","valence",
        "tempo","time_signature"
    ]

    ft_response = requests.get(track_url,headers=headers)
    ft_result = ft_response.json()

    for f in feat_names:
        v = ft_result.get(f)
        if v != None:
            track_data.append(ft_result[f])
        else:
            track_data.append(-100000)
    final_data = {}

    ind = 0
    for name in prev_names + feat_names:
        final_data[name.title()] = track_data[ind]
        ind += 1
    return True,"Success",track_data,final_data
