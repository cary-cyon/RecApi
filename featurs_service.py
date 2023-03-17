import librosa
import numpy as np

class FeatursService():
    @classmethod
    def get_features(cls,track_name) -> list:
        test_list = []
        y, sr = librosa.load(track_name)
        #chroma_stft
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        test_list.append(np.mean(chroma_stft))

        #spectral_centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        test_list.append(np.mean(spec_cent))

        #spectral_bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        test_list.append(np.mean(spec_bw))

        #rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        test_list.append(np.mean(rolloff))

        #zero_crossing_rate
        zcr = librosa.feature.zero_crossing_rate(y)
        test_list.append(np.mean(zcr))

        #mfcc 1 - 20
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        for e in mfcc:
            test_list.append(np.mean(e))
        return test_list