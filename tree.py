import numpy as np
from abc import ABC, abstractmethod


class Node:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.split_left = None
        self.split_right = None

    @property
    def n(self):
        return len(self.X)

    def get_split(self, split_index: np.ndarray) -> tuple[Node, Node]:
        left = Node(self.X[split_index], self.y[split_index])
        right = Node(self.X[~split_index], self.y[~split_index])
        return left, right

    def apply_split(self, split_index: np.ndarray) -> tuple[Node, Node]:
        left, right = self.get_split(split_index)
        self.split_left = left
        self.split_right = right
        return left, right


class Tree:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.root_node = Node(X, y)


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
    def __init__(self, criterion: SplitCriterion, min_samples: int):
        self.criterion = criterion

    def split(self, node: Node) -> tuple[Node, Node] | None:
        split_index = self.best_split(node)
        if not split_index:
            return None
        return node.get_split(split_index)

    def split_cost(self, node: Node, split_index: np.ndarray) -> float:
        left, right = node.get_split(split_index)
        # Weighted average of costs
        return (self.splitter.cost(left) * left.n + self.splitter.cost(right) * right.n) / node.n

    def best_split(self, node: Node) -> np.ndarray | None:
        base_cost = self.splitter.cost(node)
        if base_cost == 0:
            return None

        split_costs = [(split, self.split_cost(node, split)) for split in self.generate_splits(node.X)]
        best_split, min_cost = min(split_costs, key=lambda split_cost: split_cost[1])
        if min_cost > base_cost:
            return None
        return best_split

    @abstractmethod
    def generate_splits(self, X: np.ndarray) -> list[np.ndarray]:
        pass


class TreeBuilder(ABC):
    def __init__(self, splitter: Splitter, min_samples: float = 0.1):
        self.splitter = splitter
        self.min_samples
