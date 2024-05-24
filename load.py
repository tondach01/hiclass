import zipfile
from os import sep, remove
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import handle_nan


class Dataset:
    """
    Represents ARFF dataset for hierarchical classification.

    Requires directory structure as follows:

        load.py
        datasets_FUN -> XXX_FUN -> XXX_FUN.{train,test,valid}.arff.zip
    """

    def __init__(self, dataset_name: str, nan_strategy: str = "mean", args=None):
        """
        Create Dataset object, consisting of training/testing/validation data

        :param dataset_name: name of the dataset (without _FUN suffix) - one of {cellcycle, church, derisi, eisen,
        expr, gasch1, gasch2, hom, pheno, seq, spo, struc}
        :param nan_strategy: strategy to be used for NaN values - one of "mean", "knn", "remove". If not
        provided or not one of allowed, "mean" is used
        :param args: possible dictionary of arguments to NaN-handling functions

        the hom_FUN dataset is quite large and takes a lot of time to process
        struc_FUN takes moderate amount of time (around 5 minutes on my laptop)
        """
        path = sep.join(["datasets_FUN", f"{dataset_name}_FUN"])

        def _read(which: str) -> pd.DataFrame:
            file = f"{dataset_name}_FUN.{which}.arff"
            p = sep.join([path, file+".zip"])
            zipfile.ZipFile(p).extract(file)

            # Apparently, scipy cannot read hierarchical attributes
            def _read_arff(f: str) -> pd.DataFrame:
                attr_names = []
                with open(f) as arff_file:
                    reading_attrs = True
                    types = {"class": "object"}
                    # gasch1 dataset has two columns of the same name
                    used_names, i = set(), 0
                    while reading_attrs:
                        attr = arff_file.readline().strip().split()
                        if (not attr) or (attr[0].upper() != "@ATTRIBUTE"):
                            if attr and attr[0].upper() == "@DATA":
                                reading_attrs = False
                            continue
                        if attr[1] in used_names:
                            attr[1] += f"_{i}"
                            i += 1
                        attr_names.append(attr[1])
                        used_names.add(attr[1])
                        if attr[2].startswith("{"):
                            types[attr[1]] = "category"

                    d = pd.read_csv(arff_file, names=attr_names, na_values=["?"], dtype=types)

                    # might be a problem when "mean" imputing one-hot encoded category with > 2 levels
                    categories = [column for column, t in types.items() if t == "category"]
                    if categories:
                        enc = OneHotEncoder(sparse_output=False)
                        encoded_columns = enc.fit_transform(d[categories])
                        encoded_df = pd.DataFrame(encoded_columns,
                                                  columns=enc.get_feature_names_out(categories))
                        d = pd.concat([d.drop(columns=categories), encoded_df], axis=1)

                    d["class"] = d["class"].map(lambda x: [label for label in x.split("@")])

                    # touches also test data, but since HiClass classifiers cannot handle NaN values, this has to be
                    # done anyway
                    if nan_strategy == "remove" and which != "test":
                        d = handle_nan.remove_nan(d)
                    elif nan_strategy == "knn":
                        k = 5
                        if args is not None:
                            k = args.get("k", 5)
                        d = handle_nan.impute_knn(d, k)
                    else:
                        d = handle_nan.impute_mean(d)

                return d

            data = _read_arff(file)
            remove(file)
            return data

        self.train = _read("train")
        self.test = _read("test")
        self.valid = _read("valid")

    def _x(self, data: pd.DataFrame, expand: bool = False) -> pd.DataFrame:
        df = data.copy()
        if expand:
            df = self.expand_multi_class(df)
        df.pop("class")
        return df

    def _y(self, data: pd.DataFrame, expand: bool = False) -> pd.Series:
        df = data.copy()
        if expand:
            df = self.expand_multi_class(df)
            return df["class"].apply(lambda x: x.split("/"))
        return df["class"].apply(lambda x: list(map((lambda y: y.split("/")), x)))

    def x_train(self, expand: bool = False) -> pd.DataFrame:
        """
        Get features (not classes) of examples in training part of dataset

        :param expand: whether to expand multi-label rows to multiple records
        :return: copy of dataframe with classes removed
        """
        return self._x(self.train, expand)

    def y_train(self, expand: bool = False) -> pd.Series:
        """
        Get classes (not features) of examples in training part of dataset

        :param expand: whether to expand multi-label rows to multiple records
        :return: copy of dataframe with features removed
        """
        return self._y(self.train, expand)

    def x_test(self, expand: bool = False) -> pd.DataFrame:
        """
        Get features (not classes) of examples in test part of dataset

        :param expand: whether to expand multi-label rows to multiple records
        :return: copy of dataframe with classes removed
        """
        return self._x(self.test, expand)

    def y_test(self, expand: bool = False) -> pd.Series:
        """
        Get classes (not features) of examples in test part of dataset

        :param expand: whether to expand multi-label rows to multiple records
        :return: copy of dataframe with features removed
        """
        return self._y(self.test, expand)

    def x_valid(self, expand: bool = False) -> pd.DataFrame:
        """
        Get features (not classes) of examples in validation part of dataset

        :param expand: whether to expand multi-label rows to multiple records
        :return: copy of dataframe with classes removed
        """
        return self._x(self.valid, expand)

    def y_valid(self, expand: bool = False) -> pd.Series:
        """
        Get classes (not features) of examples in validation part of dataset

        :param expand: whether to expand multi-label rows to multiple records
        :return: copy of dataframe with features removed
        """
        return self._y(self.valid, expand)

    @staticmethod
    def expand_multi_class(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand the dataset so that each row has just one class label

        :param df: dataset to be expanded
        :return: the same dataset, but multi-label rows are duplicated for each label
        """
        return df.explode("class", ignore_index=True)
