import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


def impute_mean(x: pd.DataFrame):
    imp = SimpleImputer(strategy="mean").fit(x)
    return pd.DataFrame(imp.transform(x), columns=imp.feature_names_in_)


def impute_knn(x: pd.DataFrame, k=5):
    imp = KNNImputer(n_neighbors=k).fit(x)
    return pd.DataFrame(imp.transform(x), columns=imp.feature_names_in_)


def remove_nan(x: pd.DataFrame, y: pd.DataFrame):
    nan_rows = x[x.isnull().T.any()]
    return x.drop(labels=nan_rows), y.drop(labels=nan_rows)
