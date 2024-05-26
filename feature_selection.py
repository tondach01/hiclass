import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import (
    SelectorMixin,
    mutual_info_classif,
    SelectKBest,
)
from math import sqrt, floor
from sklearn.tree import DecisionTreeClassifier
from hiclass.MultiLabelLocalClassifierPerNode import (
    MultiLabelLocalClassifierPerNode,
)
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
    if isinstance(y, pd.Series):
        max_len = y.apply(len).agg(max)
        depth = y.apply(lambda x: max([len(label) for label in x])).agg(max)
    else:
        max_len = max(map(len, y))
        depth = max(map(lambda x: max(map(len, x)), y))

    return np.array([
        [
            list(label) + [""] * (depth - len(label))
            for label in row
        ] + [
            [""] * depth
        ] * (max_len - len(row))
        for row in y
    ])


class ModSelectKBest(SelectorMixin, BaseEstimator):
    """
    Perform "flat" selection of k best parameters based on mutual information. The hierarchy is ignored and labels
    worked with as strings

    Sample usage:
        import load
        import feature_selection
        from sklearn.tree import DecisionTreeClassifier
        from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode
        from hiclass.metrics import f1

        dataset = load.Dataset("cellcycle", nan_strategy="mean")
        x_train, y_train = dataset.x_train(), dataset.y_train()
        x_test, y_test = dataset.x_test(), dataset.y_test()

        tree = DecisionTreeClassifier()
        classifier = MultiLabelLocalClassifierPerNode(local_classifier=tree)

        selector = ModSelectKBest().fit(x_train, y_train)
        x_train = selector.transform(x_train)

        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(selector.transform(x_test))
        print(f1(fill_reshape(y_test), y_pred))
        """
    def __init__(self, *, k=10, sqrt_features=False):
        """
        Create ModSelectKBest object

        :param k: number of features to be chosen
        :param sqrt_features: take square root of number of features as k
        """
        self.k = k
        self.sqrt_features = sqrt_features

    def set_params(self, k=10, sqrt_features=False) -> 'ModSelectKBest':
        self.k = k
        self.sqrt_features = sqrt_features
        return self

    def fit(self, x, y):
        self.n_features_in_ = x.shape[1]

        y_pd = pd.DataFrame(y, columns=["class"])
        x_exp = self._expand_multi_class(pd.concat([x, y_pd], axis=1))
        y_exp = x_exp["class"].copy()
        x_exp.drop(columns="class", inplace=True)

        y_exp = y_exp.map(lambda label: "/".join(label))
        if self.sqrt_features:
            self.k = floor(sqrt(x.shape[1]))
        self.selector_ = SelectKBest(mutual_info_classif, k=self.k).fit(x_exp, y_exp)
        self.feature_names_in_ = x.columns
        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.selector_.get_support(False)

    def _more_tags(self):
        return {"requires_y": True}

    @staticmethod
    def _expand_multi_class(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand the dataset so that each row has just one class label

        :param df: dataset to be expanded
        :return: the same dataset, but multi-label rows are duplicated for each label
        """
        return df.explode("class", ignore_index=True)


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

    Sample usage:
        import load
        import feature_selection
        from sklearn.tree import DecisionTreeClassifier
        from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode
        from hiclass.metrics import f1

        dataset = load.Dataset("cellcycle", nan_strategy="mean")

        x_train, y_train = dataset.x_train(), dataset.y_train()
        x_valid, y_valid = dataset.x_valid(), dataset.y_valid()
        x_test, y_test = dataset.x_test(), dataset.y_test()

        tree = DecisionTreeClassifier()
        classifier = MultiLabelLocalClassifierPerNode(local_classifier=tree)

        feats = feature_selection.iterative_select(x_train, y_train, x_valid, y_valid, sqrt_features=True, r_seed=42)

        classifier.fit(x_train.get(feats), y_train)

        y_pred = classifier.predict(x_test.get(feats))
        print(f1(fill_reshape(y_test), y_pred))
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

        if isinstance(y_valid, pd.Series):
            y_valid = fill_reshape(y_valid)

        score = f1(y_valid, y_pred)

        if score > f1_best:
            f1_best = score
            sample_best = s

        print(f"Epoch {i+1}/{epochs}: F1 score "
              f"on validation set {round(score, 5)}",
              flush=True)

    return sample_best


class IterativeSelect(SelectorMixin, BaseEstimator):
    def __init__(self,
                 *,
                 x_valid: pd.DataFrame,
                 y_valid: pd.DataFrame,
                 k=10,
                 sqrt_features=False,
                 epochs=10,
                 r_seed=None,
                 verbose=False):
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.k = k
        self.sqrt_features = sqrt_features
        self.epochs = epochs
        self.r_seed = r_seed
        self.verbose = verbose

    def set_params(self,
                   *,
                   x_valid: pd.DataFrame,
                   y_valid: pd.DataFrame,
                   k=10,
                   sqrt_features=False,
                   epochs=10,
                   r_seed=None) -> 'IterativeSelect':
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.k = k
        self.sqrt_features = sqrt_features
        self.epochs = epochs
        self.r_seed = r_seed

    def _get_support_mask(self):
        check_is_fitted(self)

        mask = np.zeros(len(self.x_valid.columns), dtype=bool)
        mask[self.sample_best_] = True
        return mask

    def fit(self, x, y) -> 'IterativeSelect':
        self.n_features_in_ = x.shape[1]

        if self.sqrt_features:
            self.k = floor(sqrt(x.shape[1]))

        if self.r_seed is not None:
            seed(self.r_seed)

        y_valid_reshaped = fill_reshape(self.y_valid)

        f1_best = 0
        self.sample_best_ = []

        for i in range(self.epochs):
            s = np.zeros(self.n_features_in_, dtype=bool)
            s[np.random.choice(self.n_features_in_,
                               self.k,
                               replace=False)] = True

            tree = DecisionTreeClassifier()
            classifier = MultiLabelLocalClassifierPerNode(
                local_classifier=tree
            )

            classifier.fit(x.loc[:, s], y)

            y_pred = classifier.predict(self.x_valid.loc[:, s])
            score = f1(y_valid_reshaped, y_pred)

            if score > f1_best:
                f1_best = score
                self.sample_best_ = s

            if self.verbose:
                print(f"Epoch {i+1}/{self.epochs}: F1 score "
                      f"on validation set {round(score, 5)}",
                      flush=True)

        return self

    def _more_tags(self):
        return {"requires_y": True}
