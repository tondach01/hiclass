import handle_nan
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from math import sqrt


def unique_labels(y):
    labels = set()
    for row in y:
        for label in row:
            labels.add(" ".join(label))
    return [label.split() for label in labels]


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

    def is_leaf(self):
        return not self.children

    def get_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        return None

    def print_node(self, prefix="", last=True):
        print(prefix + ("\u2514 " if last else "\u251c ") + self.name)
        for i, child in enumerate(self.children):
            if i == len(self.children) - 1:
                child.print_node(prefix + ("  " if last else "\u2502 "), last=True)
            else:
                child.print_node(prefix + ("  " if last else "\u2502 "), last=False)


class Hierarchy:
    def __init__(self, labels):
        self.root = Node("root")
        for label in labels:
            current = self.root
            for level in label:
                child = current.get_child(level)
                if child is None:
                    child = Node(level)
                    child.parent = current
                    current.children.append(child)
                current = child

    def print_hierarchy(self):
        self.root.print_node()


def select_k_best(x: pd.DataFrame, y: pd.DataFrame, k=10, sqrt_features: bool = False) -> list:
    """
    Perform "flat" selection of k best parameters based on mutual information. The hierarchy is ignored and labels
    worked with as strings

    :param x: features
    :param y: labels
    :param k: number of features to be chosen
    :param sqrt_features: take square root of number of features as k
    :return: names of selected features
    """
    y = y.map(lambda label: "/".join(label))
    if sqrt_features:
        k = sqrt(x.shape[0])
    selector = SelectKBest(mutual_info_classif, k=k).fit(x, y)
    return selector.get_feature_names_out(input_features=x.columns)
