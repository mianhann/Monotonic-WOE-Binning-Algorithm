import pandas as pd
import pytest

from monotonic_binning.monotonic_woe_binning import Binning

###
# pytest fixtures
###

################################################################################################################


@pytest.fixture(scope="session")
def testing_data_train():
    # Data available at https://online.stat.psu.edu/stat508/resource/analysis/gcd
    train = pd.read_csv("data/Training50.csv")
    test = pd.read_csv("data/Test50.csv")
    return train


@pytest.fixture(scope="session")
def testing_data_test():
    # Data available at https://online.stat.psu.edu/stat508/resource/analysis/gcd
    test = pd.read_csv("data/Test50.csv")
    return test


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


def test_generate_summary(testing_data_train):
    """To avoid repeating class init, fixtures etc. should be used."""
    # Data available at https://online.stat.psu.edu/stat508/resource/analysis/gcd
    # train = pd.read_csv("data/Training50.csv")
    # test = pd.read_csv("data/Test50.csv")
    # replace the above data with a fixture
    train = testing_data_train
    ###
    # Specify variables
    var = "Age..years."  # variable to be binned
    y = "Creditability"  # the target variable
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
    bin_object.dataset = train
    bin_object.column = bin_object.dataset.columns[
        bin_object.dataset.columns != bin_object.y
    ][0]
    bin_object.generate_summary()

    assert isinstance(bin_object.init_summary, pd.DataFrame)
    assert ~bin_object.init_summary.empty


def test_combine_bins():
    pass


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
