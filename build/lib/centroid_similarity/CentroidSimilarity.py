import numpy as np
from scipy.stats import f as fdist
from multitest import MultiTest
import logging
from .MultiClassStats import MultiClassStats

class CentroidSimilarity(object):
    """
    Classify high-dimensional data based on nearest class average
    
    Fitting can be done by extracting parameters from a fitted MultiClassStats 
    object. The latter can be trained in a streaming setting. 
    
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=[]):
        if type(X) == MultiClassStats:
            self.classes_ = X.get_class_names()
            
            nums = np.expand_dims(np.array(list(X.num.values())), 1)
            
            self._cls_n = np.squeeze(nums)
            m = np.vstack(list(X.sum.values()))
            self._cls_mean = m / nums
            ss = np.vstack(list(X.sum_squares.values()))
            self._cls_std = np.sqrt(ss / (nums - 1))
            
            self._global_mean = np.mean(m / nums, 0)
            self._global_std = np.sqrt( np.mean((ss + nums * (self._cls_mean - self._global_mean) ** 2) / (nums - 1), 0) )

        else: 
            self._fit(X, y)
            
        self.set_mat(np.ones_like(self._cls_mean))
        
    def _fit(self, X, y):
        self.classes_ = np.unique(y)
        self._global_mean = np.mean(X, 0)
        self._global_std = np.std(X, 0)
        
        self._cls_mean = np.zeros((len(self.classes_), X.shape[1]))
        self._cls_std = np.zeros((len(self.classes_), X.shape[1]))
        #self._cls_n = np.zeros((len(self.classes_), X.shape[1]))
        self._cls_n = np.zeros(len(self.classes_))
        
        for i,_ in enumerate(self.classes_):
            X_cls = X[y == self.classes_[i]]
            self._cls_mean[i] = np.mean(X_cls, 0)
            self._cls_std[i] = np.std(X_cls, 0)
            self._cls_n[i] = len(X_cls)
        
        
    def set_mat(self, mask):
        means = self._cls_mean * mask
        self._mask = mask
        self._mat = (means.T / np.linalg.norm(means, axis=1)).T
        
    def logprob_func(self, response):
        """
        Overide this function according to probability model 
        to report on true log-probabilities
        """
        
        return response
    
    def get_centroids(self):
        return self._mat * self._mask
        
    def predict_log_proba(self, X):
        response = X @ self.get_centroids().T
        return self.logprob_func(response)
    
    def predict(self, X):
        probs = self.predict_log_proba(X)
        return np.argmax(probs, 1)

class CentroidSimilarityFeatureSelection(CentroidSimilarity):
    
    def fit(self, X, y=[], method='one_vs_all'):
        
        super().fit(X, y)
        self._cls_response = np.zeros(len(self.classes_))
        mask = np.ones_like(self._cls_mean)
        
        for i, cls in enumerate(self.classes_):
            mask[i] = self.get_cls_mask(i, method=method)
        
        if mask.sum() == 0:
            logging.warn("non of the coordinates were found to be informative. ",
                             "Ignoring mask.")
        else:
            self.set_mat(mask)
        
    
    def get_pvals(self, cls_id, method='one_vs_all'):
        """
        compute P-values associated with each feature
        for the given class
        """
        
        mu1 = self._cls_mean[cls_id]
        n1 = self._cls_n[cls_id]
        std1 = self._cls_std[cls_id]
        nG = self._cls_n.sum(0)
        stdG = self._global_std
        muG = self._global_mean

        assert(method in ['one_vs_all', 'diversity_pursuit'])
        if method == 'one_vs_all' :
            pvals,_,_ = one_vs_all_ANOVA(n1, nG, mu1, muG, std1, stdG)
        if method == 'diversity_pursuit':
            pvals,_,_ = diversity_pursuit_ANOVA(self._cls_n,
                                                self._cls_mean,
                                                self._cls_std)
        return pvals

    
    def get_cls_mask(self, cls_id, method='one_vs_all'):
        """
        compute class feature mask
        """        
        pvals = self.get_pvals(cls_id, method=method)
        
        mt = MultiTest(pvals)
        hc, hct = mt.hc_star(gamma=.2)
        self._cls_response[cls_id] = hc
        mask = pvals < hct
        
        return mask
            
        
def diversity_pursuit_ANOVA(nn, mm, ss):
    """
    F-test for discoverying discriminating features
    
    The test is vectorized along the last dimention where
    different entires corresponds to different features
    
    Args:
    -----
    :nn:  vector indicating the number of elements in each class
    :mm:  matrix of means; the (i,j) entry is the mean response of
          class i in feature j
    :ss:  matrix of standard errors; the (i,j) entry is the standard
          error of class i in feature j
    
    """
    ###
    nn = np.expand_dims(nn, 1)
    ###
    
    muG = np.sum(nn * mm, 0) / np.sum(nn, 0) # global mean
    SSres = np.sum(nn * (mm - muG) ** 2, 0)
    SSfit = np.sum(nn * (ss ** 2), 0)
    #SSerr = SStot - SSfit

    dfn = len(nn) - 1
    dfd = np.sum(nn, 0) - len(nn)

    F = ( SSres / dfn ) / ( SSfit / dfd )
    return fdist.sf(F, dfn, dfd), SSres, SSfit
        
        
def one_vs_all_ANOVA(n1, nG, mu1, muG, std1, stdG):
    """
    Equivalent to two-sample t-test
    
    Uses the global mean and std to evaluate Group2,
    obtained by removing Group1 from the global mean
    and std.
    
    Args:
    -----
    
    Return:
    ------
    pvalue, SSbetween, SSbetween

    """
    n2 = nG - n1
    mu2 = (muG * nG - mu1 * n1) / (nG - n1)
    SSres = n1 * (mu1 - muG) ** 2 + n2 * (mu2 - muG) ** 2
    SStot = stdG ** 2 * nG
    SSfit = SStot - SSres 

    F = ( SSres / 1 ) / ( SSfit / (nG - 2) )
    return fdist.sf(F, 1, nG - 2), SSres, SSfit
