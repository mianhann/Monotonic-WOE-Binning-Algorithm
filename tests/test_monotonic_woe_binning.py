import numpy as np
import pandas as pd
import pytest

from monotonic_binning.monotonic_woe_binning import Binning

np.random.seed(42)

###
# pytest fixtures
# Data available at https://online.stat.psu.edu/stat508/resource/analysis/gcd
###

################################################################################################################


@pytest.fixture(scope="session")
def testing_data_train():

    train = pd.read_csv("data/Training50.csv")

    return train


@pytest.fixture(scope="session")
def testing_data_test():

    test = pd.read_csv("data/Test50.csv")

    return test


@pytest.fixture(scope="session")
def testing_data_train_bootstrap(n):

    train = pd.read_csv("data/Training50.csv")
    data_bs = train.sample(n=n, replace=True)

    return data_bs


@pytest.fixture(scope="session")
def testing_data_test_bootstrap(n):

    test = pd.read_csv("data/Test50.csv")
    data_bs = test.sample(n=n, replace=True)

    return data_bs


################################################################################################################

###
# bin_object1
###
@pytest.fixture(scope="session")
def bin_object1():

    train = pd.read_csv("data/Training50.csv")
    # Specify variables
    var = "Age..years."  # variable to be binned
    y = "Creditability"  # the target variable
    n_threshold = 50
    y_threshold = 10
    p_threshold = 0.35
    # Create binning object for testing
    bin_object1 = Binning(
        y,
        n_threshold=n_threshold,
        y_threshold=y_threshold,
        p_threshold=p_threshold,
        sign=False,
    )
    bin_object1.dataset = train
    bin_object1.column = bin_object1.dataset.columns[
        bin_object1.dataset.columns != bin_object1.y
    ][0]

    return bin_object1


################################################################################################################

###
# pytest tests
###


def test_initialization():
    # Specify variables
    y = "target_var"  # the target variable
    n_threshold = 50
    y_threshold = 10
    p_threshold = 0.35
    # Create binning object for testing
    bin_object = Binning(
        y,
        n_threshold=n_threshold,
        y_threshold=y_threshold,
        p_threshold=p_threshold,
        sign=False,
    )
    # Property assertions
    assert bin_object.y == y
    assert bin_object.n_threshold == n_threshold
    assert bin_object.y_threshold == y_threshold
    assert bin_object.p_threshold == p_threshold
    assert bin_object.sign == False


def test_generate_summary(bin_object1):
    bin_object1.generate_summary()

    assert isinstance(bin_object1.init_summary, pd.DataFrame)
    assert ~bin_object1.init_summary.empty


def test_combine_bins(bin_object1):
    assert ~bin_object1.bin_summary.empty


def test_calculate_pvalues():
    pass


def test_calculate_woe():
    pass


def test_generate_bin_labels():
    pass


def test_generate_final_dataset():
    pass


def test_fit():
    pass


def test_transform():
    pass


if __name__ == "__main__":
    print("Running tests")
