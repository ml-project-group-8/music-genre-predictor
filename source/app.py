from flask import Flask, render_template, redirect, send_from_directory
from flask import jsonify, request
import os

from get_lyrics import grab_lyrics,grab_features

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/songdata",methods=["POST"])
def songdata():
    print(request.values)
    query = request.form.get("q", "", type=str)
    if query == "":
        return jsonify({"error":"invalid query"})

    lyrical_data = grab_lyrics(query)
    if lyrical_data[0] == False:
        return jsonify({"error":"unable to grab lyrics"})
    feats = grab_features(query)
    if feats[0] == False:
        return jsonify({"error":"unable to grab features"})


    return jsonify({"lyrics":lyrical_data[2],"features":feats[-1]})
if __name__ == "__main__":
    app.run(debug=True)
