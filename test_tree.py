from tree import Tree, TreeBuilder, QuantileSplitter, Gini, MSE, ConditionalQuantileSplitter
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=125)


def gen_simple():
    X = np.array([0, 1, 2, 3]).reshape((-1, 1))
    y = np.array([0, 0, 1, 1])
    return X, y


def gen_battenburg():
    X = rng.random((1000, 2))
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)) | ((X[:, 0] <= 0.5) & (X[:, 1] <= 0.5))
    return X, y


def gen_presence():
    X = rng.random((1000, 3))
    X[:, 0] = np.round(X[:, 0])
    for i in range(1, X.shape[1]):
        X[:, i] = np.where(X[:, 0] == 0, 0.0, X[:, i])
    y = X[:, 1:].sum(axis=1)
    return X, y

name = "std"
# name = "conditional"

# criterion = Gini()
criterion = MSE()

splitter = QuantileSplitter(criterion)
# splitter = ConditionalQuantileSplitter(criterion, conditions = {0: [1, 2]})

builder = TreeBuilder(splitter, min_samples=10)

X, y = gen_presence()

tree = Tree(builder)
tree.fit(X, y)

# score = accuracy_score(np.round(tree.predict(X)), y)
score = r2_score(tree.predict(X), y)

X_test = np.array([[0] * 101, np.linspace(0, 1, 101), [0] * 101]).T
preds = tree.predict(X_test)

exp = shap.Explainer(tree.predict, X, feature_names=["Present", "Area", "Attention"])
explanations = exp(X[:100])
plot = shap.plots.beeswarm(explanations, show=False)
plt.gcf().tight_layout()
plt.gcf().savefig(f"{name}-beeswarm.png")
