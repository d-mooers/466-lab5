#!/usr/bin/env python3
# CSC 466
# Fall 2021
# Lab 3
# Dylan Mooers - dmooers@calpoly.edu
# Justin Huynh - jhuynh42@calpoly.edu
# Purpose - Runs cross validation on random forest


import pandas as pd
import argparse
from pathlib import Path
from Forest import RandomForest


def parse():
    parser = argparse.ArgumentParser(description="Random Forest Algorithm")
    parser.add_argument(
        "trainingSetFile", type=str, help="name of csv file containing dataset"
    )
    parser.add_argument(
        "--t",
        type=float,
        nargs="?",
        default=0.01,
        help="Threshold for building the tree; default is 0",
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="?",
        default=10,
        help="Number of folds for cross-validation; default is 0",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="?",
        default=3,
        help="Number of attributes each decision tree is built on; default is 3",
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="?",
        default=10,
        help="Number of decision trees to build; default is 10",
    )
    parser.add_argument(
        "--NumDataPoints",
        type=int,
        nargs="?",
        default=10,
        help="Number of data points each decision tree is built on; default is 10",
    )

    parser.add_argument("--s", default=False, action="store_true", help="silent mode")

    args = vars(parser.parse_args())
    return args


def main():
    args = parse()
    training_fname = args["trainingSetFile"]
    k = args["k"]
    m = args["m"]
    numDataPoints = args["NumDataPoints"]
    N = args["N"]

    threshold = args["t"]
    silent = args["s"]

    tmp = pd.read_csv(training_fname)

    # header = list(tmp)
    # domain = tmp.iloc[[0]].values.tolist()[0]
    # classAttr = tmp.iloc[[1]].values.tolist()[0][0]
    # skip = [i for i, j in zip(header, domain) if int(j) < 0]
    # numeric = [i for i, j in zip(header, domain) if int(j) == 0]
    classAttr = "_Author_"

    # tmp.drop(tmp.index[[0]], inplace=True)
    # tmp.drop(tmp.index[[0]], inplace=True)
    # tmp.drop(columns=skip)
    # skip2 = tmp[(tmp == "?").any(axis=1)]
    # tmp.drop(skip2.index, axis=0, inplace=True)
    # tmp.reset_index(drop=True, inplace=True)
    attributes = list(tmp.head())
    numeric = [i for i in attributes if i != classAttr]
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.max_seq_items = None
    pd.options.display.width = 0
    forest = RandomForest(tmp, classAttr, threshold, attributes, numeric, m, numDataPoints, N)
    forest.buildForest(tmp)
    testing = tmp.drop(columns=classAttr)
    classifications = forest.classifyMany(tmp)
    classifications.to_csv("rfAuthorship.out")



if __name__ == "__main__":
    main()
