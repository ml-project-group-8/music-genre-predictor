import pandas as pd
import requests
import base64
import pprint
import time
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
    print("Auth request hase status code {}".format(code))
    if "access_token" not in data:
        print("Could not login. Exiting")
        return
    token = data["access_token"]
    return token
def main():

    tracks = []

    playlist_url = "https://api.spotify.com/v1/playlists/{}/tracks"
    category_url = "https://api.spotify.com/v1/browse/categories/{}/playlists"

    cat_data  = {
        "market": "US",
        "limit": 50, # This is a max we gotta iterate to get them all
        "offset": 0, # Initially 0, we udate this until no more playlist
    }
    play_data = {
        "market": "US",
        "limit": 100, # This is a max we gotta iterate to get them all
        "offset": 0, # Initially 0, we udate this until no more playlist
    }
    categories = ["rnb","hiphop","country","pop","classical","edm_dance","rock"]
    start   = time.time()
    token   = get_token()

    headers = {
        "Authorization": "Bearer {}".format(token),
    }
    for cat in categories:
        cat_count = 0
        cat_data  = {
            "market": "US",
            "limit": 50, # This is a max we gotta iterate to get them all
            "offset": 0, # Initially 0, we udate this until no more playlist
        }
        weGood = False
        while cat_count < 2000 and not weGood:
            print("Getting playlists for {}. Count: {}".format(cat,cat_count))

            cat_url = category_url.format(cat)

            if time.time() - start > 3598:
                time.sleep(5)
                start = time.time()
                token = get_token()

            cat_response = requests.get(cat_url, headers=headers, params=cat_data)

            cat_playlists = cat_response.json()
            print(cat_playlists)
            time.sleep(10)
            if "error" in cat_playlists:
                print("Error, {}".format(cat_playlists["error"]))
                return

            for playlist in cat_playlists["playlists"]["items"]:
                curr_playlist = playlist_url.format(playlist["id"])

                if time.time() - start > 3598:
                    time.sleep(5)
                    start = time.time()
                    token = get_token()

                pl_response = requests.get(curr_playlist,headers=headers,params=play_data)
                pl_result = pl_response.json()

                if "error" in pl_result:
                    print("Error, {}".format(pl_result))

                for song in pl_result.get("items"):
                    cat_count += 1
                    track_data = song.get("track")

                    if not track_data:
                        continue

                    track_id   = track_data["id"]           # Id code for track
                    track_name = track_data["name"]         # Title of the song
                    track_pop  = track_data["popularity"]   # How popular a song is
                    track_exp  = track_data["explicit"]     # Bool of cussing or nah
                    track_gen  = cat[::]                    # Category
                    track_art = track_data["artists"][0]["name"] # First artist on track

                    tracks.append( [track_gen,track_id, track_pop, track_exp, track_name,track_art] )
            cat_data["offset"] += 50

    print("Got IDs for {} songs".format(len(tracks)))

    features   = []
    feat_names = [
        "danceability","energy","key","loudness","mode","speechiness",
        "acousticness","instrumentalness","liveness","valence",
        "tempo","time_signature"
    ]
    index = 0
    for track in tracks:
        if index % 150 == 0:
            print(index)
        track_url    = "https://api.spotify.com/v1/audio-features/{}".format(track[1])
        analysis_url = "https://api.spotify.com/v1/audio-analysis/{}".format(track[1])

        if time.time() - start > 3598:
            time.sleep(5)
            start = time.time()
            token = get_token()

        tr_response = requests.get(track_url,headers=headers)
        tr_result = tr_response.json()


        if "error" in tr_result:
            print("Error, {}".format(tr_result["error"]))
            if tr_result["error"]["status"] == 429:
                while  "error" in tr_result and tr_result["error"]["status"] == 429:
                    time.sleep(10)
                    tr_response = requests.get(track_url,headers=headers)
                    tr_result = tr_response.json()
            else:
                continue

        curr   = list(track)

        for f in feat_names:
            v = tr_result.get(f)
            if v != None:
                curr.append(tr_result[f])
            else:
                curr.append(-100000)
        album_url = "https://api.spotify.com/v1/albums/{}".format(track[0])

        features.append(curr)
        index+= 1
    df = pd.DataFrame(features,columns=["Genre","ID","popularity","is_exp","name","artist"]+feat_names)
    df.to_csv("final.csv")






if __name__ == "__main__":
    main()
