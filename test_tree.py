#!/usr/bin/env python

import sys
from tree import Tree, TreeBuilder, QuantileSplitter, Gini, MSE, ConditionalQuantileSplitter
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
import sklearn
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt
import subprocess
import plotly.express as px

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


def get_tree(name):
    if name == "sklearn":
        return sklearn.tree.DecisionTreeRegressor(max_depth=3, min_samples_leaf=9)

    criterion = MSE()
    if name == "std":
        splitter = QuantileSplitter(criterion)
    else:
        splitter = ConditionalQuantileSplitter(criterion, conditions = {0: [1, 2]})

    builder = TreeBuilder(splitter, min_samples=10, max_depth=3)
    return Tree(builder)


name = sys.argv[1]
print(name)
tree = get_tree(name)
X, y = gen_presence()
tree.fit(X, y)

feature_names = ["Present", "Area", "Attention"]

if name != "sklearn":
    tree.export_graphviz(name + ".dot", feature_names=feature_names)
else:
    sklearn.tree.export_graphviz(tree, name + ".dot", feature_names=feature_names)
subprocess.call(["dot", "-Tpng", "-Gdpi=300", name + ".dot", "-o", name + ".png"])

# score = accuracy_score(np.round(tree.predict(X)), y)
score = r2_score(tree.predict(X), y)
print(f"{score=:.3f}")

for present, present_name in (0, "present-false"), (1, "present-true"):
    for i, feature_name in list(enumerate(feature_names))[1:]:
        X_test = [[present] * 101, [0] * 101, [0] * 101]
        X_test[i] = np.linspace(0, 1, 101)
        X_test = np.array(X_test).T
        preds = tree.predict(X_test)
        fig = px.line(x=X_test[:, i], y=preds).update_layout(xaxis_title=feature_name, yaxis_title="Predicted target")
        fig.write_image(f"{name}-dependence-{present_name}-{feature_name}.png")

exp = shap.Explainer(tree.predict, X, feature_names=feature_names)
explanations = exp(X[:100])
plot = shap.plots.beeswarm(explanations, show=False)
plt.gcf().tight_layout()
plt.gcf().savefig(f"{name}-beeswarm.png")
