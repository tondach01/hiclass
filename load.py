import zipfile
from os import sep, remove
import pandas as pd
import numpy as np
import handle_nan


class Dataset:
    """
    Represents ARFF dataset for hierarchical classification.

    Requires directory structure as follows:

        load.py
        datasets_FUN -> XXX_FUN -> XXX_FUN.{train,test,valid}.arff.zip
    """

    def __init__(self, dataset_name: str, expand: bool = False, nan_strategy: str = "mean", args=None):
        """
        Create Dataset object, consisting of training/testing/validation data

        :param dataset_name: name of the dataset (without _FUN suffix)
        :param expand: whether to expand multi-class rows
        :param nan_strategy: strategy to be used for NaN values - one of "mean", "knn", "remove". If not
        provided or not one of allowed, "mean" is used
        :param args: possible dictionary of arguments to NaN-handling functions

        dataset_name: one of {cellcycle, church, derisi, eisen, expr, gasch1, gasch2, hom, pheno, seq, spo, struc}

        the hom_FUN dataset is quite large and takes a lot of time to process
        struc_FUN takes moderate amount of time (around 5 minutes on my laptop)
        """
        path = sep.join(["datasets_FUN", f"{dataset_name}_FUN"])
        self.expand = expand

        def _read(which: str):
            file = f"{dataset_name}_FUN.{which}.arff"
            p = sep.join([path, file+".zip"])
            zipfile.ZipFile(p).extract(file)
            data = _read_arff(file)
            remove(file)
            return data

        # Apparently, scipy cannot read hierarchical attributes
        def _read_arff(file: str):
            attr_names = []
            with open(file) as arff_file:
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

                data = pd.read_csv(arff_file, names=attr_names, na_values=["?"], dtype=types)
                data["class"] = data["class"].map(lambda x: [label for label in x.split("@")])
                if nan_strategy == "remove":
                    data = handle_nan.remove_nan(data)
                elif nan_strategy == "knn":
                    k = args.get("k", 5)
                    data = handle_nan.impute_knn(data, k)
                else:
                    data = handle_nan.impute_mean(data)

            return data

        self.train = _read("train")
        self.test = _read("test")
        self.valid = _read("valid")

    def _x(self, data: pd.DataFrame):
        df = data.copy()
        if self.expand:
            df = self.expand_multi_class(df)
        df.pop("class")
        return df

    def _y(self, data: pd.DataFrame):
        df = data.copy()
        if self.expand:
            df = self.expand_multi_class(df)
            return df["class"].apply(lambda x: x.split("/"))
        return df["class"].apply(lambda x: list(map((lambda y: y.split("/")), x)))

    def x_train(self):
        """
        Get features (not classes) of examples in training part of dataset

        :return: copy of dataframe with classes removed
        """
        return self._x(self.train)

    def y_train(self):
        """
        Get classes (not features) of examples in training part of dataset

        :return: copy of dataframe with features removed
        """
        return self._y(self.train)

    def x_test(self):
        """
        Get features (not classes) of examples in test part of dataset

        :return: copy of dataframe with classes removed
        """
        return self._x(self.test)

    def y_test(self):
        """
        Get classes (not features) of examples in test part of dataset

        :return: copy of dataframe with features removed
        """
        return self._y(self.test)

    def x_valid(self):
        """
        Get features (not classes) of examples in validation part of dataset

        :return: copy of dataframe with classes removed
        """
        return self._x(self.valid)

    def y_valid(self):
        """
        Get classes (not features) of examples in validation part of dataset

        :return: copy of dataframe with features removed
        """
        return self._y(self.valid)

    @staticmethod
    def expand_multi_class(df: pd.DataFrame):
        """
        Expand the dataset so that each row has just one class label

        :param df: dataset to be expanded
        :return: the same dataset, but multi-label rows are duplicated for each label
        """
        return df.explode("class", ignore_index=True)
