import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


def impute_mean(x: pd.DataFrame) -> pd.DataFrame:
    """
    Impute NaN values by simple column mean

    :param x: data to be imputed
    :return: new DataFrame with values imputed
    """
    imp = SimpleImputer(strategy="mean").fit(x)
    return pd.DataFrame(imp.transform(x), columns=imp.feature_names_in_)


def impute_knn(x: pd.DataFrame, k=5) -> pd.DataFrame:
    """
    Impute NaN values by KNN algorithm

    :param x: data to be imputed
    :param k: number of nearest neighbors
    :return: new DataFrame with values imputed
    """
    imp = KNNImputer(n_neighbors=k).fit(x)
    return pd.DataFrame(imp.transform(x), columns=imp.feature_names_in_)


def remove_nan(x: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """
    Remove rows with NaN values from the data

    :param x: data features, possibly containing NaN values
    :param y: corresponding labels
    :return: features and labels without rows containing NaN
    """
    nan_rows = x[x.isnull().T.any()]
    return x.drop(labels=nan_rows), y.drop(labels=nan_rows)
