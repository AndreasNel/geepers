import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataManager:
    def __init__(self):
        self.dataset = pd.DataFrame()
        self.training_set = pd.DataFrame()
        self.validation_set = pd.DataFrame()
        self.testing_set = pd.DataFrame()
        self.fieldnames = []

    def load_dataset(self):
        """
        Loads the data set from the files and formats it. This function is responsible for setting the fieldnames
        and the dataset properties of the data manager.
        :return: self
        """
        # Read the different field names
        self.fieldnames = pd.read_csv('field_names.csv', header=None, usecols=[0], squeeze=True).tolist()
        # Read the data, with the appropriate headings
        dataset = pd.read_csv('data_set.csv', header=None, names=self.fieldnames)
        replacements = [False if t in ['unknown', 'normal'] else True for t in dataset.attack_type.unique()]
        dataset.attack_type.replace(dataset.attack_type.unique(), replacements, inplace=True)

        self.fieldnames = self.fieldnames[:-1]
        dataset.drop(columns=['difficulty_level'], inplace=True)

        le = LabelEncoder()
        dataset.protocol_type = le.fit_transform(dataset.protocol_type)
        dataset.service = le.fit_transform(dataset.service)
        dataset.flag = le.fit_transform(dataset.flag)

        dataset[dataset.columns.difference(['attack_type'])] = dataset[
            dataset.columns.difference(['attack_type'])].astype(float)
        dataset[['protocol_type', 'service', 'flag']] = dataset[['protocol_type', 'service', 'flag']].astype(int)
        self.dataset = dataset
        return self

    def split_dataset(self, train_ratio, test_ratio):
        """
        Splits the loaded data set into a training, validation and testing set based on the given ratios.
        :param train_ratio: The ratio of the training set in relation to the size of the data set.
        :param test_ratio: The ratio of the testing set in relation to the size of the data set.
        :return: self
        """
        self.training_set, self.testing_set, self.validation_set = np.split(
            self.dataset.sample(frac=1.0),
            [int(train_ratio * len(self.dataset)), int((train_ratio + test_ratio) * len(self.dataset))])
        return self

    def get_training_set(self):
        return self.training_set

    def get_testing_set(self):
        return self.testing_set

    def get_validation_set(self):
        return self.validation_set

    def get_data_set(self):
        return self.dataset

    def get_field_names(self):
        return self.fieldnames
