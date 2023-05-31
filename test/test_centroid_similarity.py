import sys
import numpy as np
sys.path.append('../')
from centroid_similarity.CentroidSimilarity import CentroidSimilarityFeatureSelection, CentroidSimilarity

c = 2
n = 100
d = 1000

mu = 5 / np.sqrt( n ) # signal strength

eps = .01 # non sparsity rate

sparsity_pattern = np.random.rand(d) < eps
contrast_vector = mu * sparsity_pattern

y = np.random.randint(c,size = n)
X = (np.expand_dims(2*(y - 1/2), 1) * contrast_vector) + np.random.randn(n, d) 

test_train_split_ratio = .2
train_indices = np.random.rand(n) > test_train_split_ratio

X_train, X_test = X[train_indices], X[~train_indices]
y_train, y_test = y[train_indices], y[~train_indices]

clf_naive = CentroidSimilarity()
clf_fs = CentroidSimilarityFeatureSelection()

clf_naive.fit(X_train, y_train)
clf_fs.fit(X_train, y_train, method='one_vs_all')
y_pred_naive = clf_naive.predict(X_test)
y_pred_fs = clf_fs.predict(X_test)
print("Accuracy (naive) = ", np.mean(y_pred_naive == y_test))
print("Accuracy (feature selection) = ", np.mean(y_pred_fs == y_test))
