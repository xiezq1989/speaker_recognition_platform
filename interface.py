import pickle
from collections import defaultdict
from skgmm import GMMSet
from features import get_feature
import time,os
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

    def mfcc_dump(self, fname):
        """ dump all features to file"""
        with open(fname, 'wb') as f:
            pickle.dump(self.features, f, -1)

    def train(self):
        self.gmmset = GMMSet()
        start_time1 = time.time()
        print("Begin to train")
        for name, feats in self.features.items():
            try:
                start_time2 = time.time()
                self.gmmset.fit_new(feats, name)
                print(name," trained",time.time() - start_time2, "seconds" )
            except Exception as e :
                print ("%s failed because of %s"%(name,e))
        print ("Train ",time.time() - start_time1, " seconds")

    def dump(self, save_dir):
        """ dump all models to file"""
        # 每个GMM模型独立保存一个模型文件
        for i in range(len(self.gmmset.y)):
            label=self.gmmset.y[i]
            model=self.gmmset.gmms[i]
            file_name=label+'.m'
            save_path=os.path.join(save_dir,file_name)
            with open(save_path, 'wb') as f:
                # 这里保存的是skgmm.GMMSet object
                pickle.dump(model, f, -1)
        #self.gmmset.after_pickle()

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
        #return self.gmmset.predict_one(feat)
        return self.predict_one(feat)

    @staticmethod
    def load(fname):
        """ load from a dumped model file"""
        with open(fname, 'rb') as f:
            label = os.path.basename(fname.rstrip('/')).split('.')[0]
            R = pickle.load(f)
            #R.gmmset.after_pickle()
            return label,R
