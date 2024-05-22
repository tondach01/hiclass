import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


def impute_mean(x):
    imp = SimpleImputer(strategy="mean").fit(x)
    return pd.DataFrame(imp.transform(x), columns=imp.feature_names_in_)


def impute_knn(x, k=5):
    imp = KNNImputer(n_neighbors=k)
    return imp.fit_transform(x)
