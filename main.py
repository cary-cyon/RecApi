import os
from flask import Flask, jsonify
from flask import request, abort
import pandas as pd
from recomendation_service import RecomendationService
from featurs_service import FeatursService
from ClassifierService import ClassifierService

app = Flask(__name__)
data = pd.read_csv("res_data_for_songs.csv")
recSys = RecomendationService()
features_list = ['chroma_stft','spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1',
                'mfcc2', 'mfcc3', 'mfcc4',  'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
                'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']
genres_list = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

@app.route("/tracks", methods =["GET"])
def get_list_tracks():
    title = request.args.get('title')
    col = int(request.args.get('col'))
    if col == None or col <= 0:
        col = 5
    if title == None:
        return abort(400)
    list = recSys.GetRecomendation(title, data, col)
    return jsonify(list)


@app.route("/features", methods = ["POST"])
def post_return_track_features():
    track = request.files['track1']
    track.save(os.path.join("upload", track.filename))
    list_of_features = FeatursService.get_features(os.path.join("upload", track.filename))
    list_of_features = list(map(str, list_of_features))
    json_dict = dict(zip(features_list, list_of_features))

    return jsonify(json_dict)

@app.route("/genres", methods = ["POST"])
def post_return_track_geners():
    track = request.files['track1']
    track.save(os.path.join("upload", track.filename))
    list_of_features = FeatursService.get_features(os.path.join("upload", track.filename))
    model = ClassifierService('model.txt')
    res = model.get_ganres_list(list_of_features)[0]
    return jsonify(dict(zip(genres_list, res)))

@app.route("/track2tracks", methods =["POST"])
def post_return_res_of_traks():
    track = request.files['track1']
    track.save(os.path.join("upload", track.filename))
    list_of_features = FeatursService.get_features(os.path.join("upload", track.filename))
    model = ClassifierService('model.txt')
    res = model.get_ganres_list(list_of_features)[0]
    list_of_rec = recSys.GetRecondationByVector(res, data, 10)
    return jsonify(list_of_rec)


if __name__ == "__main__":
    app.run()