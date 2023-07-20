import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin

DESIRED_WIDTH = 320
pd.set_option("display.width", DESIRED_WIDTH)
pd.set_option("display.max_columns", 130)
warnings.filterwarnings("ignore")
os.getcwd()


class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, y, n_threshold, y_threshold, p_threshold, sign=False):

        self.n_threshold, self.y_threshold, self.p_threshold = (
            n_threshold,
            y_threshold,
            p_threshold,
        )
        self.y = y
        self.sign = sign

        self.init_summary = pd.DataFrame()
        self.bin_summary = pd.DataFrame()
        self.pvalue_summary = pd.DataFrame()
        self.dataset = pd.DataFrame()
        self.woe_summary = pd.DataFrame()

        self.column = object
        self.total_iv = object
        self.bins = object
        self.bucket = object

    def generate_summary(self):

        self.init_summary = (
            self.dataset.groupby([self.column])
            .agg({self.y: ["mean", "std", "size"]})
            .rename({"mean": "means", "size": "nsamples", "std": "std_dev"}, axis=1)
        )

        self.init_summary.columns = self.init_summary.columns.droplevel(level=0)

        self.init_summary = self.init_summary[["means", "nsamples", "std_dev"]]
        self.init_summary = self.init_summary.reset_index()

        self.init_summary["del_flag"] = 0
        self.init_summary["std_dev"] = self.init_summary["std_dev"].fillna(0)

        self.init_summary = self.init_summary.sort_values(
            [self.column], ascending=self.sign
        )

    def combine_bins(self):

        summary = self.init_summary.copy()
        while True:
            i = 0
            summary = summary[summary.del_flag != 1]
            summary = summary.reset_index(drop=True)
            while True:
                j = i + 1
                if j >= len(summary):
                    break
                if summary.iloc[j].means < summary.iloc[i].means:
                    i = i + 1
                    continue
                else:
                    while True:
                        n = summary.iloc[j].nsamples + summary.iloc[i].nsamples
                        m = (
                            summary.iloc[j].nsamples * summary.iloc[j].means
                            + summary.iloc[i].nsamples * summary.iloc[i].means
                        ) / n
                        if n == 2:
                            s = np.std([summary.iloc[j].means, summary.iloc[i].means])
                        else:
                            s = np.sqrt(
                                (
                                    summary.iloc[j].nsamples
                                    * (summary.iloc[j].std_dev ** 2)
                                    + summary.iloc[i].nsamples
                                    * (summary.iloc[i].std_dev ** 2)
                                )
                                / n
                            )
                        summary.loc[i, "nsamples"] = n
                        summary.loc[i, "means"] = m
                        summary.loc[i, "std_dev"] = s
                        summary.loc[j, "del_flag"] = 1
                        j = j + 1
                        if j >= len(summary):
                            break
                        if summary.loc[j, "means"] < summary.loc[i, "means"]:
                            i = j
                            break
                if j >= len(summary):
                    break
            dels = np.sum(summary["del_flag"])
            if dels == 0:
                break
        self.bin_summary = summary.copy()

    @staticmethod
    def calculate_leads(df, column_names=("nsamples", "means", "std_dev")):

        column_names_lead = [col + "_lead" for col in column_names]
        df = df.assign(
            **{
                column_names_lead[0]: df[column_names[0]].shift(-1),
                column_names_lead[1]: df[column_names[1]].shift(-1),
                column_names_lead[2]: df[column_names[2]].shift(-1),
            }
        )

        return df

    @staticmethod
    def calculate_estimations(df, column_names=("nsamples", "means", "std_dev")):

        df["est_" + column_names[0]] = (
            df[column_names[0] + "_lead"] * df[column_names[0]]
        )
        df["est_" + column_names[1]] = (
            df[column_names[0] + "_lead"] * df[column_names[1] + "_lead"]
            + df[column_names[0]] * df[column_names[1]]
        ) / df["est_" + column_names[0]]
        df["est_" + column_names[2] + "2"] = (
            df[column_names[0] + "_lead"] * df[column_names[2] + "_lead"] ** 2
            + df[column_names[0]] * df[column_names[2]] ** 2
        ) / (df["est_" + column_names[0]] - 2)

        return df

    @staticmethod
    def calculate_p_value(df, column_names=("nsamples", "means", "std_dev")):

        df = Binning.calculate_leads(df)
        df = Binning.calculate_estimations(df)
        df["z_value"] = (
            (df[column_names[1]] - df[column_names[1] + "_lead"])
            / np.sqrt(df["est_" + column_names[2] + "2"])
            * (1 / df[column_names[0]] + 1 / df[column_names[0] + "_lead"])
        )
        df["p_value"] = 1 - stats.norm.cdf(df["z_value"])

        return df

    def calculate_pvalues(self):

        summary = self.bin_summary.copy()
        while True:
            summary = (
                summary.pipe(self.calculate_leads)
                .pipe(self.calculate_estimations)
                .pipe(self.calculate_p_value)
            )
            mask = (
                (summary["nsamples"] < self.n_threshold)
                | (summary["nsamples_lead"] < self.n_threshold)
                | (summary["means"] * summary["nsamples"] < self.y_threshold)
                | (summary["means_lead"] * summary["nsamples_lead"] < self.y_threshold)
            )
            summary["p_value"] = np.where(
                mask, summary["p_value"] + 1, summary["p_value"]
            )

            max_p = np.max(summary["p_value"].values)
            row_of_maxp = summary["p_value"].idxmax()
            row_delete = row_of_maxp + 1

            if max_p > self.p_threshold:
                summary = summary.drop(summary.index[row_delete])
                summary = summary.reset_index(drop=True)
            else:
                break

            summary["means"] = np.where(
                summary["p_value"] == max_p, summary["est_means"], summary["means"]
            )

            summary["nsamples"] = np.where(
                summary["p_value"] == max_p,
                summary["est_nsamples"],
                summary["nsamples"],
            )

            summary["std_dev"] = np.where(
                summary["p_value"] == max_p, summary["est_std_dev2"], summary["std_dev"]
            )

        self.pvalue_summary = summary.copy()

    def calculate_woe(self):

        woe_summary = self.pvalue_summary[[self.column, "nsamples", "means"]]

        woe_summary["bads"] = (
            woe_summary["means"].values * woe_summary["nsamples"].values
        )
        woe_summary["goods"] = (
            woe_summary["nsamples"].values - woe_summary["bads"].values
        )

        total_goods = np.sum(woe_summary["goods"].values)
        total_bads = np.sum(woe_summary["bads"].values)

        woe_summary["dist_good"] = woe_summary["goods"].values / total_goods
        woe_summary["dist_bad"] = woe_summary["bads"].values / total_bads

        woe_summary["WOE_" + self.column] = np.log(
            woe_summary["dist_good"].values / woe_summary["dist_bad"].values
        )

        woe_summary["IV_components"] = (
            woe_summary["dist_good"].values - woe_summary["dist_bad"].values
        ) * woe_summary["WOE_" + self.column].values

        self.total_iv = np.sum(woe_summary["IV_components"].values)
        self.woe_summary = woe_summary

    def generate_bin_labels(self, row):

        return "-".join(
            map(str, np.sort([row[self.column], row[self.column + "_shift"]]))
        )

    def generate_final_dataset(self):

        if self.sign == False:
            shift_var = 1
            self.bucket = True
        else:
            shift_var = -1
            self.bucket = False

        self.woe_summary[self.column + "_shift"] = self.woe_summary[self.column].shift(
            shift_var
        )

        if self.sign == False:
            self.woe_summary.loc[0, self.column + "_shift"] = -np.inf
            self.bins = np.sort(list(self.woe_summary[self.column]) + [np.Inf, -np.Inf])
        else:
            self.woe_summary.loc[
                len(self.woe_summary) - 1, self.column + "_shift"
            ] = np.inf
            self.bins = np.sort(list(self.woe_summary[self.column]) + [np.Inf, -np.Inf])

        self.woe_summary["labels"] = self.woe_summary.apply(
            self.generate_bin_labels, axis=1
        )

        self.dataset["bins"] = pd.cut(
            self.dataset[self.column], self.bins, right=self.bucket, precision=0
        )

        self.dataset["bins"] = self.dataset["bins"].astype(str)
        self.dataset["bins"] = self.dataset["bins"].map(
            lambda x: x.lstrip("[").rstrip(")")
        )

    def fit(self, dataset):

        self.dataset = dataset
        # only two columns expected, the one that is not the target is the variable to be binned
        self.column = self.dataset.columns[self.dataset.columns != self.y][0]

        self.generate_summary()
        self.combine_bins()
        self.calculate_pvalues()
        self.calculate_woe()
        self.generate_final_dataset()

    def transform(self, test_data):

        test_data[self.column + "_bins"] = pd.cut(
            test_data[self.column], self.bins, right=self.bucket, precision=0
        )

        return test_data
