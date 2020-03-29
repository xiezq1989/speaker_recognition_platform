#from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
import operator
import numpy as np
import math

class GMMSet:

    def __init__(self, gmm_order = 32):
        self.gmms = []
        self.gmm_order = gmm_order
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GaussianMixture(self.gmm_order)
        gmm.fit(x)
        self.gmms.append(gmm)

    def gmm_score(self, gmm, x):
        return np.sum(gmm.score(x))

    @staticmethod
    def softmax(scores):
        score_exp  = [math.exp(i) for i in scores]
        scores_sum  = sum(score_exp)
        #scores_sum = sum([math.exp(i) for i in scores])
        #score_max  = math.exp(max(scores))
        softmax_scores=[score_exp[n]/scores_sum for n in range(len(score_exp))]
        return softmax_scores 
        #return round(score_exp / scores_sum, 3)

    def predict_one(self, x):
        scores = [self.gmm_score(gmm, x) / len(x) for gmm in self.gmms]
        #print("scores:",scores)
        #softmax_scores = self.softmax(scores) 
        result = [(self.y[index], value) for (index, value) in enumerate(scores)]
        #print("result:",result)
        #sort_res = sorted(result, key=operator.itemgetter(1), reverse=True)
        #p = max(result, key=operator.itemgetter(1))
        #print("Top 5:")
        #print(sort_res[:5])
        #return sort_res[:5]
        return result

    def before_pickle(self):
        pass

    def after_pickle(self):
        pass
