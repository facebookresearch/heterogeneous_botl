"""
This file contains the HPOBHandler class, which is modified from the
original provided in https://github.com/releaunifreiburg/HPO-B under
the MIT license that is included in this directory.

To run HPO-B benchmarks, the datasets must be downloaded to
`hpob-data` sub-directory. Please refer to the GitHub repo for
the download link.
"""

import numpy as np

try:
    import ujson as json
except Exception:
    import json
import os

this_dir = os.path.abspath(os.path.dirname(__file__))


class HPOBHandler:

    def __init__(
        self,
        root_dir=os.path.join(this_dir, "hpob-data/"),
        mode="v3",
        # surrogates_dir="saved-surrogates/",
    ):
        """
        Constructor for the HPOBHandler.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * mode: mode name indicating how to load the data. Options:
                - v1: Loads HPO-B-v1
                - v2: Loads HPO-B-v2
                - v3: Loads HPO-B-v3
                - v3-test: Loads only the meta-test split from HPO-B-v3
                - v3-train-augmented: Loads all splits from HPO-B-v3, but augmenting
                    the meta-train data with the less frequent search-spaces.
            * surrogates_dir: path to directory with surrogates models.

        """

        print("Loading HPO-B handler")
        self.mode = mode
        # self.surrogates_dir = surrogates_dir
        self.seeds = ["test0", "test1", "test2", "test3", "test4"]

        if self.mode == "v3-test":
            self.load_data(root_dir, only_test=True)
        elif self.mode == "v3-train-augmented":
            self.load_data(root_dir, only_test=False, augmented_train=True)
        elif self.mode in ["v1", "v2", "v3"]:
            self.load_data(root_dir, version=self.mode, only_test=False)
        else:
            raise ValueError("Provide a valid mode")

        # surrogates_file = surrogates_dir + "summary-stats.json"
        # if os.path.isfile(surrogates_file):
        #     with open(surrogates_file) as f:
        #         self.surrogates_stats = json.load(f)

    def load_data(
        self, rootdir="", version="v3", only_test=True, augmented_train=False
    ):
        """
        Loads data with some specifications.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * version: name indicating what HPOB version to use. Options: v1, v2, v3).
            * Only test: Whether to load only testing data (valid only for version v3).
                Options: True/False
            * augmented_train: Whether to load the augmented train data (valid only for
                version v3). Options: True/False

        """

        print("Loading data...")
        meta_train_augmented_path = os.path.join(
            rootdir, "meta-train-dataset-augmented.json"
        )
        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir, "meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
        bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

        with open(meta_test_path, "rb") as f:
            self.meta_test_data = json.load(f)

        with open(bo_initializations_path, "rb") as f:
            self.bo_initializations = json.load(f)

        if not only_test:
            if augmented_train or version == "v1":
                with open(meta_train_augmented_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            else:
                with open(meta_train_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            with open(meta_validation_path, "rb") as f:
                self.meta_validation_data = json.load(f)

        if version != "v3":
            temp_data = {}
            for search_space in self.meta_train_data.keys():
                temp_data[search_space] = {}

                for dataset in self.meta_train_data[search_space].keys():
                    temp_data[search_space][dataset] = self.meta_train_data[
                        search_space
                    ][dataset]

                if search_space in self.meta_test_data.keys():
                    for dataset in self.meta_test_data[search_space].keys():
                        temp_data[search_space][dataset] = self.meta_test_data[
                            search_space
                        ][dataset]

                    for dataset in self.meta_validation_data[search_space].keys():
                        temp_data[search_space][dataset] = self.meta_validation_data[
                            search_space
                        ][dataset]

            self.meta_train_data = None
            self.meta_validation_data = None
            self.meta_test_data = temp_data

        self.search_space_dims = {}

        for search_space in self.meta_test_data.keys():
            dataset = list(self.meta_test_data[search_space].keys())[0]
            X = self.meta_test_data[search_space][dataset]["X"][0]
            self.search_space_dims[search_space] = len(X)

    def normalize(self, y, y_min=None, y_max=None):

        if y_min is None:
            return (y - np.min(y)) / (np.max(y) - np.min(y))
        else:
            return (y - y_min) / (y_max - y_min)

    def evaluate(
        self,
        bo_method,
        search_space_id,
        dataset_id,
        seed,
        n_trials=10,
    ):
        """
        Evaluates a method on the benchmark with discretized search-spaces.
        Inputs:
            * bo_method: object to evaluate. It should have a function (class method)
                named 'observe_and_suggest'.
            * search_space_id: Identifier of the search spaces for the evaluation.
                Option: see original paper.
            * dataset_id: Identifier of the dataset for the evaluation.
                Options: see original paper.
            * seed: Identifier of the seed for the evaluation. Options: test0, test1,
                test2, test3, test4.
            * trails: Number of trials (iterations on the optimization).
        Ooutput:
            * a list with the maximumu performance (incumbent) for every trial.

        """
        assert hasattr(
            bo_method, "observe_and_suggest"
        ), "The provided  object does not have a method called ´observe_and_suggest´"

        try:
            X = np.array(self.meta_test_data[search_space_id][dataset_id]["X"])
            y = np.array(self.meta_test_data[search_space_id][dataset_id]["y"])
        except KeyError:
            print(self.meta_test_data.keys())
            raise
        y = self.normalize(y)
        data_size = len(X)

        pending_evaluations = list(range(data_size))
        current_evaluations = []

        init_ids = self.bo_initializations[search_space_id][dataset_id][seed]

        for i in range(len(init_ids)):
            idx = init_ids[i]
            pending_evaluations.remove(idx)
            current_evaluations.append(idx)

        max_accuracy_history = [np.max(y[current_evaluations])]
        for i in range(n_trials):
            idx = bo_method.observe_and_suggest(
                X_obs=X[current_evaluations],
                y_obs=y[current_evaluations],
                X_pen=X[pending_evaluations],
            )
            idx = pending_evaluations[idx]
            pending_evaluations.remove(idx)
            current_evaluations.append(idx)
            max_accuracy_history.append(np.max(y[current_evaluations]))

            if max(y) in max_accuracy_history:
                break

        max_accuracy_history += [max(y).item()] * (n_trials - i - 1)

        return max_accuracy_history

    def evaluate_continuous(
        self,
        bo_method,
        search_space_id,
        dataset_id,
        seed,
        n_trials=10,
    ):
        raise NotImplementedError(
            "This is a limited version that does not support evaluate_continuous."
        )

    def get_search_spaces(self):
        return list(self.meta_test_data.keys())

    def get_datasets(self, search_space):
        return list(self.meta_test_data[search_space].keys())

    def get_seeds(self):
        return self.seeds

    def get_search_space_dim(self, search_space):

        return self.search_space_dims[search_space]
