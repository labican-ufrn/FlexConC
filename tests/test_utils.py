from unittest import TestCase

import numpy as np

from src.utils import select_labels


class TestUtils(TestCase):
    def test_split_labels_binary_classes_equal_distribution(self):
        y_train = np.array([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ])
        X_train = [i for i in range(len(y_train))]
        labelled_percentage = .3

        np.random.RandomState(30)
        np.random.seed(42)

        selected_instances = select_labels(y_train, X_train, labelled_percentage)
        x_lbs, y_lbs = np.unique(selected_instances, return_counts=True)

        expected_output = [
            -1,  0, -1, -1, -1,
             0, -1, -1,  0, -1,
             1,  1, -1, -1, -1,
            -1, -1, -1,  1, -1,
        ]

        self.assertEqual(list(selected_instances), expected_output)
        self.assertEqual(list(x_lbs), [-1, 0, 1])
        self.assertEqual(list(y_lbs), [14, 3, 3])

    def test_split_labels_binary_classes_non_equal_distribution(self):
        y_train = np.array([
            0, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ])
        X_train = [i for i in range(len(y_train))]
        labelled_percentage = .1

        np.random.RandomState(30)
        np.random.seed(42)

        selected_instances = select_labels(y_train, X_train, labelled_percentage)
        x_lbs, y_lbs = np.unique(selected_instances, return_counts=True)

        expected_output = [
             0,  1, -1, -1, -1,
            -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1,
        ]

        self.assertEqual(list(selected_instances), expected_output)
        self.assertEqual(list(x_lbs), [-1, 0, 1])
        self.assertEqual(list(y_lbs), [18, 1, 1])

    def test_split_labels_multi_classes_not_equal_distribution(self):
        y_train = np.array([
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
        ])
        X_train = [i for i in range(len(y_train))]
        labelled_percentage = .3

        np.random.RandomState(30)
        np.random.seed(42)

        selected_instances = select_labels(
            y_train,
            X_train,
            labelled_percentage
        )
        x_lbs, y_lbs = np.unique(selected_instances, return_counts=True)

        expected_output = [
            -1,  0, -1, -1, -1,
             1, -1, -1,  1, -1,
            -1, -1, -1, -1,  1,
            -1, -1, -1, -1,  2,
        ]

        self.assertEqual(list(selected_instances), expected_output)
        self.assertEqual(list(x_lbs), [-1, 0, 1, 2])
        self.assertEqual(list(y_lbs), [15, 1, 3, 1])
