import pytest
import pandas as pd

from monotonic_binning.monotonic_woe_binning import Binning


###
# test cases? (use pytest functionality)
###

def test_initialization():
    # Data available at https://online.stat.psu.edu/stat508/resource/analysis/gcd
    train = pd.read_csv("data/Training50.csv")
    test = pd.read_csv("data/Test50.csv")
    # Specify variables
    var = "Age..years." # variable to be binned
    y_var = "Creditability" # the target variable
    # Create binning object for testing
    bin_object = Binning(y_var, n_threshold = 50, y_threshold = 10, p_threshold = 0.35, sign=False)
    # ...
    assert bin_object.y_var == y_var

def test_generate_summary():
    pass

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