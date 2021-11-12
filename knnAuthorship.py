#!/usr/bin/env python3
# CSC 466
# Fall 2021
# Lab 3
# Dylan Mooers - dmooers@calpoly.edu
# Justin Huynh - jhuynh42@calpoly.edu
# Purpose - Performs K nearest neighbors algorihtm

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import math
from collections import Counter
from Vector import Vector, parseVector


class KNN:
    def __init__(self, vectors, k, metric="cosine"):
        self.vectors = vectors
        self.k = k
        self.simMetric = metric

    def manhattan(self, a, b):
        val2 = np.sum(np.absolute(a - b))
        return val2

    def anti_dice(self, a, b):
        return np.sum(a == b) / self.numCategorical
        # matches = [i for i in a if i in b]
        # return (self.numCategorical - len(matches)) / self.numCategorical

    def normalize(self, val, attribute):
        castedVal = float(val)
        maxVal = float(self.maxValues[attribute])
        minVal = float(self.minValues[attribute])
        return (castedVal - minVal) / (maxVal - minVal)

    def get_numeric(self, row):
        return row[self.isNumeric]

    def get_categorical(self, row):
        val = row[self.isCategorical]
        return val

    def dist(self, p1, p2):
        return p1.cosine(p2)

    def _dist(self, p1, p2):
        return p1.cosine(p2)

    def process_row(self, vector):
        distances = np.vectorize(lambda x: self._dist(vector, x))(self.vectors)
        sortedIndices = np.argsort(distances)
        return [self.vectors[i].author for i in sortedIndices[-self.k:]]

    def classifyRow(self, row):
        processed = pd.Series(self.process_row(row))
        if len(processed) == 0:
            print("Learn to use a dbugger")
        val = processed.value_counts().index[0]
        return val

    def classifyAll(self, rows):
        # print('ca', rows)
        return rows.apply(self.classifyRow, axis=1)

    def knn(self):
        neighbors_and_dists = self.data.apply(lambda row: self.process_row(row), axis=1)

        res = []
        for i in list(neighbors_and_dists):
            tmp = []
            for j in i:
                tmp.append(self.data.iloc[j[0]][self.classAttr])
            res.append(tmp)
        counters = [Counter(i) for i in res]
        return [c.most_common()[0][0] for c in counters]


def parse():
    parser = argparse.ArgumentParser(description="KNN")
    parser.add_argument(
        "dataSetFile", type=str, help="name of csv file containing dataset"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="?",
        default=1,
        help="int value to determine number of neighbors to consider; default = 1",
    )
    parser.add_argument("--s", default=False, action="store_true", help="silent mode")
    parser.add_argument(
        "--metric",
        default='cos',
        nargs="?",
        type=str,
        help="Similarity metric",
    )

    args = vars(parser.parse_args())
    return args


def normalizeData(df, numericAttrs):
    df = df.astype({col: float for col in numericAttrs})
    df[numericAttrs] = df[numericAttrs].apply(
        lambda x: (x.astype(float) - min(x)) / (max(x) - min(x)), axis=0
    )
    return df


def main():
    args = parse()
    training_fname = args["dataSetFile"]
    k = args["k"]
    silent = args["s"]
    folds = -1
    metric = args['metric']

    vectors = []
    with open(training_fname, 'r') as fp:
        vectors = np.array([parseVector(line) for line in fp.readlines()])
        
    knn = KNN(vectors, k)

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.max_seq_items = None
    pd.options.display.width = 0

    # tmp = tmp.sample(frac=1)
    result = crossValidateKnn(vectors, folds, k)
    result.to_csv("knnAuthorship.out")

def crossValidateKnn(vectors, k, numNeighbors):
    if k == -1:
        k = len(vectors) - 1

    rowsPerSplit = len(vectors) // k
    result = []
    for start in range(len(vectors)):
        dropout = vectors[start]
        train = np.concatenate((vectors[:start], vectors[start + rowsPerSplit :]))
        knn = KNN(train, numNeighbors)

        prediciton = knn.classifyRow(dropout)
        result.append(prediciton)
    return pd.Series(result)


if __name__ == "__main__":
    main()
