import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from math import sqrt, floor
from sklearn.tree import DecisionTreeClassifier
from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode
from hiclass.metrics import f1
from random import sample, seed


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


def fill_reshape(y: pd.Series) -> np.ndarray:
    """
    Transform the multi-label part of the dataset to regular shape, so F1 metric can be used

    :param y: labels (not expanded)
    :return: array of x=hierarchy, y=labels per example, z=examples
    """
    max_len = y.apply(len).agg(max)
    depth = y.apply(lambda x: max([len(label) for label in x])).agg(max)

    def align(row: list):
        new = []
        for label in row:
            new.append(label + [""] * (depth-len(label)))
        return new + [[""] * depth] * (max_len-len(row))

    y = np.array(list(y.apply(align)))

    return y


def select_k_best(x: pd.DataFrame, y: pd.Series, k=10, sqrt_features: bool = False) -> list:
    """
    Perform "flat" selection of k best parameters based on mutual information. The hierarchy is ignored and labels
    worked with as strings

    :param x: features, expanded and imputed
    :param y: labels
    :param k: number of features to be chosen
    :param sqrt_features: take square root of number of features as k
    :return: names of selected features
    """
    y = y.map(lambda label: "/".join(label))
    if sqrt_features:
        k = floor(sqrt(x.shape[1]))
    selector = SelectKBest(mutual_info_classif, k=k).fit(x, y)
    return selector.get_feature_names_out(input_features=x.columns)


def iterative_select(x: pd.DataFrame, y: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series, k=10,
                     sqrt_features: bool = False, epochs: int = 10, r_seed=None) -> list:
    """
    Perform iterative selection of k best parameters based on fit to hiclass.MultiLabelLocalClassifierPerNode +
    sklearn.tree.DecisionTreeClassifier measured as F1 score.

    Selection of feature subset for each epoch is (pseudo)random, thus results may vary if seed is not specified.

    :param x: features, imputed
    :param y: labels
    :param x_valid: validation set features, imputed
    :param y_valid: validation labels
    :param k: number of features to be chosen
    :param sqrt_features: take square root of number of features as k
    :param epochs: number of turns to be taken
    :param r_seed: seed for sample generation
    :return: names of selected features
    """
    if sqrt_features:
        k = floor(sqrt(x.shape[1]))

    if r_seed is not None:
        seed(r_seed)

    columns = list(x.columns)
    f1_best = 0
    sample_best = []

    for i in range(epochs):
        s = sample(columns, k=k)
        tree = DecisionTreeClassifier()
        classifier = MultiLabelLocalClassifierPerNode(local_classifier=tree)

        classifier.fit(x.get(s), y)

        y_pred = classifier.predict(x_valid.get(s))
        score = f1(fill_reshape(y_valid), y_pred)

        if score > f1_best:
            f1_best = score
            sample_best = s

        print(f"Epoch {i+1}/{epochs}: F1 score on validation set {round(score, 5)}")

    return sample_best
