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
        
        if method == 'one_vs_all':
            for i, cls in enumerate(self.classes_):
                mask[i] = self.get_cls_mask(i)
        if method == 'diversity_pursuit':
            mask = get_mask_anova()
        
        if mask.sum() == 0:
            logging.warning("non of the coordinates were found to be informative. ",
                             "Ignoring mask.")
        else:
            self.set_mat(mask)
        
        
    def get_mask_anova(self):
        pvals,_,_ = diversity_pursuit_ANOVA(self._cls_n, self._cls_mean, self._cls_std)
        mt = MultiTest(pvals)
        hc, hct = mt.hc_star(gamma=.2)
        mask = pvals < hct
        return np.tile(mask, len(self.classes_)).T
        
    
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
            
        
def diversity_pursuit_anova(self,classes_num_samples, cls_mean, cls_std, print_shapes=False):
        """
        F-test for discovering discriminating features
        The test is vectorized along the last dimension where different entries corresponds to different features

        Args:
        -----
        :param classes_num_samples:  vector indicating the number of elements in each class
        :param cls_mean:  matrix of classes means with shape (k,p); the (i,j) entry is the class i mean in feature j
        :param cls_std:  matrix of standard errors; the (i,j) entry is the standard
              error of class i in feature j

        """
        # compute the per feature global mean - the mean of all the data for each feature. shape = (1,p)
        #global_mean = np.expand_dims(np.sum(cls_mean * classes_num_samples, 0) / np.sum(classes_num_samples, 0),axis=0)
        # compute the "between groups" sum of squares
        SSbetween = np.sum(classes_num_samples * (cls_mean - self.global_mean) ** 2, 0)
        SSwithin = np.sum((classes_num_samples-1) * (cls_std ** 2), 0)
        if print_shapes:
            print("in function diversity_pursuit_anova:")
            print(f"classes_num_samples.shape = {classes_num_samples.shape}")
            print(f"cls_std.shape = {cls_std.shape}")
            print(f"cls_mean.shape = {cls_mean.shape}")
            print(f"global_mean.shape = {self.global_mean.shape}")
            print(f"SSwithin.shape = {SSwithin.shape}")
            print(f"SSbetween.shape = {SSbetween.shape}")

        # the numerator number of degrees of freedom is actually k-1 (k is the number of groups/classes)
        dfn = len(classes_num_samples) - 1
        # the denominator number of degrees of freedom is n-k (n is the total number of samples)
        dfd = np.sum(classes_num_samples, 0) - len(classes_num_samples)
        # initialize the F statistics vector to 0 (for all features)
        f_stat = np.zeros_like(SSbetween)
        # run over all features and identify points where infinity will occur - assign an infinite F statistic there
        for i in range(SSwithin.shape[0]):
            if SSwithin[i] == 0 and SSbetween[i] > 0:
                f_stat[i] = 1e8

        f_stat = np.divide((SSbetween / dfn), (SSwithin / dfd), out=f_stat, where=(SSwithin != 0))
        return fdist.sf(f_stat, dfn, dfd), SSbetween, SSwithin


    def one_vs_all_anova(self,num_smpls_in_cls, total_num_smpls, cls_mean, global_mean, global_std):
        """
        In this method the data is partitioned to two groups: one with samples from a given class (the isolated class), and one part with the rest
        of the samples. Then, a one-way ANOVA test is performed for each feature between these two groups.
        :param num_smpls_in_cls: number of samples from the isolated class
        :param total_num_smpls: total number of samples (all classes)
        :param cls_mean: mean of the isolated class samples (basically the un-normalized class centroid)
        :param global_mean: global mean of all samples in the data
        :param global_std: standard deviation of each feature, over all samples in the data
        :return:
        """
        # get the number of samples in all other classes except the isolated class
        num_smpls_in_rest = total_num_smpls - num_smpls_in_cls
        # get the mean response in all other classes except the isolated class
        mean_smpls_in_rest = (global_mean * total_num_smpls - cls_mean * num_smpls_in_cls) / num_smpls_in_rest
        # compute the "between groups" sum of squares
        SSbetween = num_smpls_in_cls * (cls_mean - global_mean) ** 2 + num_smpls_in_rest * (mean_smpls_in_rest - global_mean) ** 2
        SStot = global_std ** 2 * (total_num_smpls - 1)
        # using the sum of squares decomposition to compute the variability within each group
        SSwithin = SStot - SSbetween
        # calculate the F-statistic
        assert (total_num_smpls > 2).all(), "total number of samples must be greater then number of groups"
        # initialize the F statistics vector to 0 (for all features)
        f_stat = np.zeros_like(SSbetween)
        # run over all features and identify points where infinity will occur - assign an infinite F statistic there
        for i in range(SSwithin.shape[0]):
            if SSwithin[i] == 0 and SSbetween[i] > 0:
                f_stat[i] = 1e8

        f_stat = np.divide(SSbetween,(SSwithin / (total_num_smpls - 2)), out=f_stat, where=(SSwithin!=0))
        return fdist.sf(f_stat, 1, total_num_smpls - 2), SSbetween, SSwithin
