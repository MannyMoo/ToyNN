from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Generator, Callable, TextIO
from sklearn.metrics import mean_squared_error
from copy import deepcopy


SplitFunction = Callable[[np.ndarray], np.ndarray]
SplitGenerator = Generator[SplitFunction, None, None]
SplitCosts = list[tuple[SplitFunction, float]]


class Node:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.split_left = None
        self.split_right = None
        self.split_function = None
        self.allowed_columns = None

    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def is_split(self) -> bool:
        return bool(self.split_left)

    @property
    def max_depth(self) -> int:
        if not self.is_split:
            return 1
        return 1 + max(self.split_left.max_depth, self.split_right.max_depth)

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
        if not self.is_split:
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

    def _graphviz_node(self, i: int, criterion: SplitCriterion, feature_names: list[str] | None) -> str:
        n = self.n
        cost = criterion.cost(self)
        label = f"nsamples = {n}\\nvalue = {self.y.mean():.3f}\\ncost = {cost:.4f}\\n"
        if self.is_split:
            isplit = self.split_function.i
            if feature_names:
                isplit = feature_names[isplit]
            else:
                isplit = f"X[{isplit}]"
            split_val = self.split_function.val
            label += f"{isplit} < {split_val:.2g}\\n"
        return f'{i} [label="{label}"] ;\n'

    def export_graphviz(
        self,
        fout: TextIO,
        criterion: SplitCriterion,
        inode: int = 0,
        inext: int = 1,
        feature_names: list[str] | None = None,
    ) -> int:
        if inode == 0:
            fout.write("""digraph Tree {
node [shape=box, fontname="helvetica"] ;
edge [fontname="helvetica"] ;
""")
            fout.write(self._graphviz_node(inode, criterion, feature_names))
        if not self.is_split:
            return inext
        ileft = inext
        fout.write(self.split_left._graphviz_node(ileft, criterion, feature_names))
        inext += 1
        iright = inext
        fout.write(self.split_right._graphviz_node(iright, criterion, feature_names))
        inext += 1
        if inode == 0:
            fout.write(f'{inode} -> {ileft} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n')
        else:
            fout.write(f"{inode} -> {ileft} ;\n")
        fout.write(f"{inode} -> {iright} ;\n")
        inext = self.split_left.export_graphviz(fout, criterion, ileft, inext, feature_names)
        inext = self.split_right.export_graphviz(fout, criterion, iright, inext, feature_names)
        return inext


class SplitCriterion(ABC):
    def cost(self, node: Node) -> float:
        return self._cost(node.y)

    @abstractmethod
    def _cost(self, y: np.ndarray) -> float:
        # Do we need X?
        pass


class Gini(SplitCriterion):
    def _cost(self, y: np.ndarray) -> float:
        n = len(y)
        gini = 1.0
        for c in np.unique(y):
            gini -= ((y == c).sum() / n) ** 2
        return gini


class MSE(SplitCriterion):
    def _cost(self, y: np.array) -> float:
        pred = y.mean()
        return mean_squared_error([pred] * len(y), y)


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
        if left.n == 0 or right.n == 0:
            # Not actually split, just return the cost of the parent node
            return self.criterion.cost(node)
        # Weighted average of costs
        return (self.criterion.cost(left) * left.n + self.criterion.cost(right) * right.n) / node.n

    def split_costs(self, node: Node) -> SplitCosts:
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


class ConditionalSplitter(Splitter):
    def __init__(self, criterion: SplitCriterion, conditions: dict[int, list[int]]):
        super().__init__(criterion)
        self.conditions = deepcopy(conditions)

    def split_costs(self, node: Node) -> SplitCosts:
        costs = super().split_costs(node)
        return [(split, cost) for split, cost in costs if split.i in node.allowed_columns]

    def split(self, node: Node) -> tuple[Node, Node] | None:
        if node.allowed_columns is None:
            node.allowed_columns = set(self.conditions)
        split_nodes = super().split(node)
        if not split_nodes:
            return None
        left, right = split_nodes
        left.allowed_columns = node.allowed_columns.copy()
        right.allowed_columns = node.allowed_columns.copy()
        i = node.split_function.i
        if i in self.conditions:
            right.allowed_columns.update(self.conditions[i])
        return left, right


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


class ConditionalQuantileSplitter(QuantileSplitter, ConditionalSplitter):
    pass


class TreeBuilder:
    def __init__(self, splitter: Splitter, min_samples: int = 10, max_depth: int = 0):
        self.splitter = splitter
        self.min_samples = min_samples
        self.max_depth = max_depth

    def build(self, root_node: Node, depth: int = 0) -> bool:
        if root_node.n < self.min_samples or (self.max_depth > 0 and depth >= self.max_depth):
            return False
        splits = self.splitter.split(root_node)
        if not splits:
            return False
        split_left, split_right = splits
        self.build(split_left, depth=depth + 1)
        self.build(split_right, depth=depth + 1)
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

    def export_graphviz(self, fout: str, feature_names: list[str] | None = None) -> None:
        with open(fout, "w") as fout:
            self.root_node.export_graphviz(
                fout, criterion=self.builder.splitter.criterion, feature_names=feature_names,
            )
            fout.write("}\n")

