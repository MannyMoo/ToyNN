from tree import Tree, TreeBuilder, QuantileSplitter, Gini
import numpy as np
from sklearn.metrics import accuracy_score

#X = np.array([0, 1, 2, 3]).reshape((-1, 1))
#y = np.array([0, 0, 1, 1])

X = np.random.random((1000, 2))
y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)) | ((X[:, 0] <= 0.5) & (X[:, 1] <= 0.5))

splitter = QuantileSplitter(Gini())
builder = TreeBuilder(splitter, min_samples=10)

tree = Tree(builder)
tree.fit(X, y)

score = accuracy_score(np.round(tree.predict(X)), y)
