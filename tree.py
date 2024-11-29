from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Generator, Callable


SplitFunction = Callable[[np.ndarray], np.ndarray]
SplitGenerator = Generator[SplitFunction, None, None]


class Node:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.split_left = None
        self.split_right = None
        self.split_function = None

    @property
    def n(self):
        return len(self.X)

    def get_split(self, split_func: SplitFunction) -> tuple[Node, Node]:
        split_index = split_func(self.X)
        left = Node(self.X[split_index], self.y[split_index])
        right = Node(self.X[~split_index], self.y[~split_index])
        return left, right

    def apply_split(self, split_func: SplitFunction) -> tuple[Node, Node]:
        left, right = self.get_split(split_func)
        self.split_left = left
        self.split_right = right
        self.split_function = split_func
        return left, right

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.split_left:
            return np.array([self.y.mean()] * len(X))

        split_index = self.split_function(X)
        preds_left = self.split_left.predict(X[split_index])
        preds_right = self.split_right.predict(X[~split_index])
        preds = []
        i_left = 0
        i_right = 0
        for isplit in split_index:
            if isplit == 1:
                preds.append(preds_left[i_left])
                i_left += 1
            else:
                preds.append(preds_right[i_right])
                i_right += 1
        return np.array(preds)


class SplitCriterion(ABC):
    def cost(self, node: Node) -> float:
        return self._cost(node.X, node.y)

    @abstractmethod
    def _cost(self, X: np.ndarray, y: np.ndarray) -> float:
        # Do we need X?
        pass


class Gini(SplitCriterion):
    def _cost(self, X: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        gini = 1.0
        for c in np.unique(y):
            gini -= ((y == c).sum() / n) ** 2
        return gini


class Splitter(ABC):
    def __init__(self, criterion: SplitCriterion):
        self.criterion = criterion

    def split(self, node: Node) -> tuple[Node, Node] | None:
        split_index = self.best_split(node)
        if split_index is None:
            return None
        return node.apply_split(split_index)

    def split_cost(self, node: Node, split_func: SplitFunction) -> float:
        left, right = node.get_split(split_func)
        # Weighted average of costs
        return (self.criterion.cost(left) * left.n + self.criterion.cost(right) * right.n) / node.n

    def split_costs(self, node: Node) -> list[tuple[SplitFunction, float]]:
        return [(split, self.split_cost(node, split)) for split in self.generate_splits(node.X)]

    def best_split(self, node: Node) -> np.ndarray | None:
        base_cost = self.criterion.cost(node)
        if base_cost == 0:
            return None

        best_split, min_cost = min(self.split_costs(node), key=lambda split_cost: split_cost[1])
        if min_cost >= base_cost:
            return None
        return best_split

    @abstractmethod
    def generate_splits(self, X: np.ndarray) -> SplitGenerator:
        pass


class SplitFunction(ABC):
    def __init__(self, i: int):
        self.i = i

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self._split_index(X[:, self.i])

    @abstractmethod
    def _split_index(self, col: np.ndarray) -> np.ndarray:
        pass


class FloatSplit(SplitFunction):
    def __init__(self, i: int, val: float):
        super().__init__(i)
        self.val = val

    def _split_index(self, col: np.ndarray) -> np.ndarray:
        return col < self.val


class QuantileSplitter(Splitter):
    def generate_splits(self, X: np.ndarray) -> SplitGenerator:
        for i in range(X.shape[1]):
            col = X[:, i]
            for qtile in np.percentile(col, np.linspace(10, 90, 9)):
                yield FloatSplit(i, qtile)


class TreeBuilder:
    def __init__(self, splitter: Splitter, min_samples: int = 10):
        self.splitter = splitter
        self.min_samples = min_samples

    def build(self, root_node: Node) -> bool:
        if root_node.n < self.min_samples:
            return False
        splits = self.splitter.split(root_node)
        if not splits:
            return False
        split_left, split_right = splits
        self.build(split_left)
        self.build(split_right)
        return True


class Tree:
    def __init__(self, builder: TreeBuilder):
        self.builder = builder
        self.root_node = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root_node = Node(X, y)
        self.builder.build(self.root_node)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.root_node.predict(X)
