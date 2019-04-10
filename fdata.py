import json
import pandas as pd
from flask import Flask, request, redirect, g, render_template
import requests
import base64
import urllib

from config import client_id, client_secret
# Authentication Steps, paramaters, and responses are defined at https://developer.spotify.com/web-api/authorization-guide/
# Visit this url to see all the steps, parameters, and expected response.


app = Flask(__name__)

#  Client Keys
CLIENT_ID = client_id
CLIENT_SECRET = client_secret

# Spotify URLS
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com"
API_VERSION = "v1"
SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)


# Server-side Parameters
CLIENT_SIDE_URL = "http://127.0.0.1"
PORT = 5000
REDIRECT_URI = "{}:{}/callback/q".format(CLIENT_SIDE_URL, PORT)
SCOPE = "playlist-modify-public playlist-modify-private"
STATE = ""
SHOW_DIALOG_bool = True
SHOW_DIALOG_str = str(SHOW_DIALOG_bool).lower()


auth_query_parameters = {
    "response_type": "code",
    "redirect_uri": REDIRECT_URI,#"http%3A%2F%2F127.0.0.1%3A5000%2Fcallback%2Fq",#REDIRECT_URI,
    "scope": SCOPE,
    # "state": STATE,
    # "show_dialog": SHOW_DIALOG_str,
    "client_id": CLIENT_ID
}

@app.route("/")
def index():
    # Auth Step 1: Authorization
    url_args = "&".join(["{}={}".format(key,urllib.parse.quote(val)) for key,val in auth_query_parameters.items()])
    query = ""

    for key in auth_query_parameters:
        query += "{}={}&".format(key, auth_query_parameters[key])

    query = query[:-1]

    auth_url = "{}/?{}".format(SPOTIFY_AUTH_URL, url_args)#query)
    return redirect(auth_url)


@app.route("/callback/q")
def callback():
    # Auth Step 4: Requests refresh and access tokens
    auth_token = request.args['code']
    code_payload = {
        "grant_type": "authorization_code",
        "code": str(auth_token),
        "redirect_uri": REDIRECT_URI,#"http%3A%2F%2F127.0.0.1%2Fcallback%2Fq"
    }
    b64val = str.encode("{}:{}".format(CLIENT_ID,CLIENT_SECRET))
    base64encoded = base64.standard_b64encode(b64val).decode("utf-8")
    headers = {"Authorization": "Basic {}".format(base64encoded)}

    post_request = requests.post(SPOTIFY_TOKEN_URL, data=code_payload, headers=headers)

    # Auth Step 5: Tokens are Returned to Application
    response_data = json.loads(post_request.text)

    access_token = response_data["access_token"]
    refresh_token = response_data["refresh_token"]
    token_type = response_data["token_type"]
    expires_in = response_data["expires_in"]

    # Auth Step 6: Use the access token to access Spotify API
    authorization_header = {"Authorization":"Bearer {}".format(access_token)}

    # Get profile data
    user_profile_api_endpoint = "{}/me".format(SPOTIFY_API_URL)
    profile_response = requests.get(user_profile_api_endpoint, headers=authorization_header)
    profile_data = json.loads(profile_response.text)

    # Get user playlist data
    playlist_api_endpoint = "{}/playlists".format(profile_data["href"])
    playlists_response = requests.get(playlist_api_endpoint, headers=authorization_header)
    playlist_data = json.loads(playlists_response.text)

    # # Combine profile and playlist data to display
    # display_arr = [profile_data] + playlist_data["items"]
    # return str(display_arr)


    # We want to grab all features playlists and get their songs. These
    # SHOULD be indicative of their genres
    playlist_url = "https://api.spotify.com/v1/browse/featured-playlists"
    token = access_token
    headers = {
        "Authorization": "Bearer {}".format(token),
    }
    data    = {
        "country": "US",
        "limit": 50, # This is a max we gotta iterate to get them all
        "offset": 0, # Initially 0, we udate this until no more playlist
    }


    p_url = "https://api.spotify.com/v1/playlists/{}/tracks"

    categories = ["rnb","hiphop","country","pop","classical","edm_dance","rock"]


    tracks = []
    for cat in categories:
        data    = {
            "market": "US",
            "limit": 50, # This is a max we gotta iterate to get them all
            "offset": 0, # Initially 0, we udate this until no more playlist
        }

        weGood = False
        cat_url = "https://api.spotify.com/v1/browse/categories/{}/playlists".format(cat)

        while not weGood:
            response = requests.get(cat_url,headers=headers,params=data)

            playlist_data = response.json()

            playlists_all = playlist_data["playlists"]["items"]
            if len(playlists_all) == 0:
                weGood=True

            temp_data    = {
                "market": "US",
                "limit": 100, # This is a max we gotta iterate to get them all
                "offset": 0, # Initially 0, we udate this until no more playlist
            }
            for plist in playlist_data["playlists"]["items"]:
                ident = plist["id"]
                my_url = p_url.format(ident)
                response = requests.get(my_url,headers=headers,params=temp_data)
                result = response.json()

                # print("\t{}".format(len(result["items"])))

                for item in result["items"]:
                    track = item["track"]
                    if not track["track"]:
                        continue
                    track_id  = track["id"]
                    track_nam = track["name"]
                    track_pop = track["popularity"]
                    track_exp = track["explicit"]
                    track_gen = cat[::]
                    try:
                        track_art = track["artists"][0]["name"]
                    except:
                        print("artist not found")
                    tracks.append( [track_gen,track_id, track_pop, track_exp, track_nam,track_art] )

                if len(result["items"]) == 100:
                    temp_data["offset"] += len(result["items"])
                else:
                    weGood = True


        features   = []
        feat_names = [
            "danceability","energy","key","loudness","mode","speechiness",
            "acousticness","instrumentalness","liveness","valence",
            "tempo","time_signature"
        ]

        for track in tracks:
            track_url    = "https://api.spotify.com/v1/audio-features/{}".format(track[1])
            analysis_url = "https://api.spotify.com/v1/audio-analysis/{}".format(track[1])

            response = requests.get(track_url,headers=headers)
            result = response.json()
            curr   = list(track)

            for f in feat_names:
                v = result.get(f)
                if v != None:
                    curr.append(result[f])
                else:
                    curr.append(-100000)
            album_url = "https://api.spotify.com/v1/albums/{}".format(track[0])

            features.append(curr)

        df = pd.DataFrame(features,columns=["Genre","ID","popularity","is_exp","name","artist"]+feat_names)
        df.to_csv("please.csv")
    else:
        # df = pd.read_
        pass
    return "success"


if __name__ == "__main__":
    app.run(debug=True,port=PORT)
