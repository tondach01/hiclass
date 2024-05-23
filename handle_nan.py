import pandas as pd
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
    # TODO apply only on suitable columns
    imp = KNNImputer(n_neighbors=k).fit(x)
    return pd.DataFrame(imp.transform(x), columns=imp.feature_names_in_)


def remove_nan(x: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with NaN values from the data

    :param x: data, possibly containing NaN values
    :return: features and labels without rows containing NaN
    """
    nan_rows = x[x.isnull().T.any()]
    return x.drop(labels=nan_rows)
