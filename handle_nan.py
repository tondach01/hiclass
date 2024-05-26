from typing import Literal, Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer


def impute_mean(x: pd.DataFrame) -> pd.DataFrame:
    """
    Impute NaN values by simple column mean

    :param x: data to be imputed
    :return: new DataFrame with values imputed
    """
    numeric = x.select_dtypes(include=["number"])
    rest = x.select_dtypes(exclude=["number"])
    imp = SimpleImputer(strategy="mean").fit(numeric)
    return pd.concat([pd.DataFrame(imp.transform(numeric), columns=imp.feature_names_in_), rest], axis=1)


def impute_knn(x: pd.DataFrame, k=5) -> pd.DataFrame:
    """
    Impute NaN values by KNN algorithm

    :param x: data to be imputed
    :param k: number of nearest neighbors
    :return: new DataFrame with values imputed
    """
    numeric = x.select_dtypes(include=["number"])
    rest = x.select_dtypes(exclude=["number"])
    imp = KNNImputer(n_neighbors=k).fit(numeric)
    return pd.concat([pd.DataFrame(imp.transform(numeric), columns=imp.feature_names_in_), rest], axis=1)


# disclaimer: may remove all the rows
def remove_nan(x: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with NaN values from the data

    :param x: data, possibly containing NaN values
    :return: features and labels without rows containing NaN
    """
    nan_rows = x[x.isnull().T.any()].index
    return x.drop(labels=nan_rows)


ImputerStrategy = Union[Literal["drop"],
                        Literal["knn"],
                        Literal["mean"],
                        Literal["median"],
                        Literal["most_frequent"],
                        Literal["constant"]]


class NumericImputer(TransformerMixin, BaseEstimator):
    def __init__(self, strategy: ImputerStrategy = 'mean', **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs

    def fit(self, X, y=None):
        self.numeric_columns_ = X.select_dtypes(include=["number"]).columns
        self.rest_columns_ = X.select_dtypes(exclude=["number"]).columns

        if self.strategy == "knn":
            if "n_neighbors" not in self.kwargs:
                self.kwargs["n_neighbors"] = 5
            self.imputer_ = KNNImputer(**self.kwargs)
        elif self.strategy == "drop":
            self.imputer_ = None
        else:
            self.imputer_ = SimpleImputer(strategy=self.strategy,
                                          **self.kwargs)

        self.imputer_.fit(X[self.numeric_columns_])
        return self

    def transform(self, X):
        if self.strategy == "drop":
            return X.dropna()

        numeric = X[self.numeric_columns_]
        rest = X[self.rest_columns_]
        numeric_imputed = pd.DataFrame(self.imputer_.transform(numeric),
                                       columns=self.numeric_columns_)
        return pd.concat([numeric_imputed, rest], axis=1)
