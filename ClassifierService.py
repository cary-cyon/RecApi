
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
class ClassifierService():
    def __init__(self, path) -> None:
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def get_ganres_list(self, vector):
        vector = np.array(vector)
        vector = vector.reshape((1,25))
        scaller = StandardScaler()
        scaller.fit(vector)
        vector = scaller.transform(vector)
        return self.model.predict_proba(vector)