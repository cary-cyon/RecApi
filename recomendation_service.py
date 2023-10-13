import numpy as np

class RecomendationService:
    def _cosine_diff(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def GetRecomendation(self, title, data_set, col_of_tracks):
        res =[]
        temp = data_set['name'] == title
        x = data_set[['1', '2', '3','4', '5', '6','7', '8', '9','10']].values
        list_of_features = x[temp].T
        names = data_set[['name']].values
        for i in range(len(names)):
            res.append((str(names[i][0]), self._cosine_diff(x[i], list_of_features)))
        res.sort(key = lambda tup: -tup[1])
        list_of_rec = []
        for i in range(0,col_of_tracks):
            list_of_rec.append(res[i][0])
        return list_of_rec
    
    def GetRecondationByVector(self, vector, data_set, col_of_tracks):
        res = []
        x = data_set[['1', '2', '3','4', '5', '6','7', '8', '9','10']].values
        list_of_features = np.array(vector).T
        names = data_set[['name']].values
        for i in range(len(names)):
            res.append((str(names[i][0]), self._cosine_diff(x[i], list_of_features)))
        res.sort(key = lambda tup: -tup[1])
        list_of_rec = []
        for i in range(0, col_of_tracks):
            list_of_rec.append(res[i][0])
        return list_of_rec


