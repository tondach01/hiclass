import zipfile
from os import sep, remove
import pandas as pd


class Dataset:
    """
    Represents ARFF dataset for hierarchical classification.

    Requires directory structure as follows:

        load.py
        datasets_FUN -> XXX_FUN -> XXX_FUN.{train,test,valid}.arff.zip
    """

    def __init__(self, dataset_name: str, expand: bool = False):
        """
        Create Dataset object, consisting of training/testing/validation data

        :param dataset_name: name of the dataset (without _FUN suffix)
        :param expand: whether to expand multi-class rows

        One of {cellcycle, church, derisi, eisen, expr, gasch1, gasch2, hom, pheno, seq, spo, struc}
        """
        path = sep.join(["datasets_FUN", f"{dataset_name}_FUN"])

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
                if expand:
                    data = self.expand_multi_class(data)

            return data

        self.train = _read("train")
        self.test = _read("test")
        self.valid = _read("valid")

    @staticmethod
    def _x(data: pd.DataFrame):
        df = data.copy()
        df.pop("class")
        return df

    @staticmethod
    def _y(data: pd.DataFrame):
        return data["class"].copy().apply(lambda x: x.split("/"))

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


if __name__ == "__main__":
    # the hom_FUN dataset is quite large and takes a lot of time to process
    # struc_FUN takes moderate amount of time (around 5 minutes on my laptop)
    for dataset in ["cellcycle", "church", "derisi", "eisen", "expr", "gasch1",
                    "gasch2", "hom", "pheno", "seq", "spo", "struc"]:
        d = Dataset(dataset, expand=True)
        y = d.y_train()
