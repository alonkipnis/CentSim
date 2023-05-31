import numpy as np


class MultiClassStats(object):
    """
    store, update, and merge mean and standard deviation
    of multi-dimensional data. Useful for classification
    based on LDA with the option to merge classes
    
    We assume that each sample is a vector of a fixed number of dimensions.
    
    Use method :add_vec: to store new information
    Use method :merge_classes: to merge several classes
    Used methods :get_mean: and :get_std: to get the vector of means and standard deviations
    of a certain class
    
    """
    def __init__(self):
        self.sum = {}
        self.sum_squares = {}
        self.num = {}  # class index
    
    
    def add_batch(self, Xs:list, Ys:list)->None:
        """
        Itertate over lists of Xs and Ys and add each sample
        """
        
        for X, y in zip(Xs, Ys):
            self.add_sample(X, y)
    
    
    def add_sample(self, X: np.ndarray, y)->None:
        """
        update sum, sum of squares, and number of samples 
        upon the addition of a new vector
        
        Args:
        :class_name:  the class associated with the sample
        :vec:         the value of the sample
        """
        if y not in self.num.keys():
            self.num[y] = 1
            self.sum[y] = X
            self.sum_squares[y] = np.zeros_like(X)
        else:
            n = self.num[y]
            delta = (self.num[y] * X - self.sum[y]) ** 2 / (n * (n+1))
            self.sum_squares[y] += delta
            self.sum[y] = self.sum[y] + X
            self.num[y] = n + 1
    
    
    def merge_classes(self, ys_to_merge, new_y)->int:
        if len(ys_to_merge) == 0:
            return None
        y0 = ys_to_merge[0]
        self.num[new_y] = self.num.pop(y0)
        self.sum_squares[new_y] = self.sum_squares.pop(y0)
        self.sum[new_y] = self.sum.pop(y0)
                                                 
        for y in ys_to_merge[1:]:
            self.num[new_y] += self.num.pop(y)
            self.sum_squares[new_y] += self.sum_squares.pop(y)
            self.sum[new_y] += self.sum.pop(y)
        return self.num[ney_y]
    
    def get_class_names(self)->list:
        return list(self.num.keys())
    
    
    def get_mean(self, y)->np.ndarray:
        return self.sum[y] / self.num[y]
    
    
    def get_std(self, y)->np.ndarray:
        return np.sqrt(self.sum_squares[y] / (self.num[y] + 1))
    
    
    def get_num(self, y)->int:
        return np.num[y]
                