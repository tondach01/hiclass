from sklearn.semi_supervised import SelfTrainingClassifier
from random import sample
import feature_selection
from sklearn.tree import DecisionTreeClassifier
from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode
from hiclass.metrics import f1

from load import Dataset


def create_classifier():
    """
    Create and configure a MultiLabelLocalClassifierPerNode with a self-training decision tree classifier.

    Returns:
    MultiLabelLocalClassifierPerNode: An instance of MultiLabelLocalClassifierPerNode configured
                                      with a SelfTrainingClassifier that uses a DecisionTreeClassifier
                                      as its base estimator. This setup is particularly suited for scenarios
                                      where multilabel classification is required and some of the training
                                      data might be unlabeled.
    Sample usage:
        import feature_selection
        from hiclass.metrics import f1
        from load import Dataset

        dataset = Dataset("cellcycle", nan_strategy="mean")
        x_test, y_test = dataset.x_test(), dataset.y_test()
        x_train, y_train = dataset.x_train(), dataset.y_train()

        classifier = create_classifier()

        selector = feature_selection.ModSelectKBest().fit(x_train, y_train)
        x_train = selector.transform(x_train)

        classifier.fit(x_train, remove_labels(dataset.y_train(), percentage))

        y_pred = classifier.predict(selector.transform(x_test))
        print(f1(feature_selection.fill_reshape(y_test), y_pred))
    """
    tree = DecisionTreeClassifier()
    self_train = SelfTrainingClassifier(base_estimator=tree)
    return MultiLabelLocalClassifierPerNode(local_classifier=self_train)


def remove_labels(labels, percentage):
    """
    Randomly sets a specified percentage of labels in the input list to [[-1]].

    :param labels (list of list): A list of labels, where each label itself can be a list of items.
    :param percentage (float): The fraction of labels to remove, represented as a float between 0 and 1.
                        For example, 0.2 means 20% of the labels will be set to [[-1]].

    Returns:
    list of list of list: The modified list of labels with a percentage of its elements set to [[-1]].

    Raises:
    - ValueError: If 'percentage' is not between 0 and 1.
    """
    if not (0 <= percentage <= 1):
        raise ValueError("Percentage of unlabeled data must be between 0 and 1.")
    size = len(labels)
    rand_indexes = sample(range(0, size), round(size*percentage))
    print("labels", labels)
    print("rand_indexes", rand_indexes)
    for i in rand_indexes:
        labels[i] = [[-1]]
    return labels


def train(dataset: Dataset, percentage: float):
    """
    Create and run semi-supervised learner on the dataset with the percentage of unlabeled data.

    :param dataset: An object representing the dataset.
    :param percentage: The fraction of the training data to be treated as unlabeled, expressed as a
      decimal (e.g., 1.0 for 100%, 0.2 for 20%).
    :param feats: features to be selected from data

    Sample usage:
        from load import Dataset

        train(Dataset("eisen"), 0.1)
    """
    x_test, y_test = dataset.x_test(), dataset.y_test()
    x_train, y_train = dataset.x_train(), dataset.y_train()

    classifier = create_classifier()

    selector = feature_selection.ModSelectKBest().fit(x_train, y_train)
    x_train = selector.transform(x_train)

    classifier.fit(x_train, remove_labels(dataset.y_train(), percentage))

    y_pred = classifier.predict(selector.transform(x_test))
    print(f1(feature_selection.fill_reshape(y_test), y_pred))
