import pickle
from collections import defaultdict
from skgmm import GMMSet
from features import get_feature
import time
#from VAD import VAD

class ModelInterface:

    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()
#        self.vad = VAD()

#    def init_noise(self, fs, signal):
 #       """
  #      init vad from environment noise
   #     """
    #    self.vad.init_noise(fs, signal)

#    def filter(self, fs, signal):
 #       """
  #      use VAD (voice activity detection) to filter out silence part of a signal
   #     """
    #    ret, intervals = self.vad.filter(fs, signal)
     #   orig_len = len(signal)

      #  if len(ret) > orig_len / 3:
            # signal is filtered by VAD
       #     return ret
        #return np.array([])

    def enroll(self, name, fs, signal):
        feat = get_feature(fs, signal)
        #print("feat:",feat)
        #print(len(feat))
        self.features[name].extend(feat)

    def train(self):
        self.gmmset = GMMSet()
        start_time = time.time()
        for name, feats in self.features.items():
            try:
                self.gmmset.fit_new(feats, name)
            except Exception as e :
                print ("%s failed"%(name))
        print ("Train ",time.time() - start_time, " seconds")

    def dump(self, fname):
        """ dump all models to file"""
        self.gmmset.before_pickle()
        with open(fname, 'wb') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    # def predict(self, fs, signal):
    #     """
    #     return a label (name)
    #     """
    #     try:
    #         feat = get_feature(fs, signal)
    #     except Exception as e:
    #         print (e)
    #     return self.gmmset.predict_one(feat)
    def predict(self, feat):
        """
        return a label (name)
        """
        return self.gmmset.predict_one(feat)

    @staticmethod
    def load(fname):
        """ load from a dumped model file"""
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R
