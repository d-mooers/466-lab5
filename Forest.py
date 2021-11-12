#!/usr/bin/env python3
# CSC 466
# Fall 2021
# Lab 3
# Dylan Mooers - dmooers@calpoly.edu
# Justin Huynh - jhuynh42@calpoly.edu
# Purpose - Builds and runs a Random Forest model

import pandas as pd
import numpy as np
from C45 import C45, Classifier


def selectData(df, classAttribute, numAttributes, numData):
    data = df.sample(numData, replace=True)
    # print(data)
    tmp = data.sample(numAttributes, replace=False, axis=1)
    tmp[classAttribute] = data[classAttribute]
    # print("---" * 10)
    # print(tmp)
    return tmp


class RandomForest:
    def __init__(
        self, data, classAttributeName, threshold, attributes, numericAttrs, m, k, N
    ):
        self.attributes = attributes.copy()
        self.c45 = C45(data, classAttributeName, threshold, attributes, numericAttrs)
        self.classAttributeValues = data[classAttributeName].unique()
        self.df = data
        self.root = None
        self.classAttributeName = classAttributeName
        self.threshold = threshold
        self.numericAttrs = numericAttrs
        self.trees = []
        self.m = m
        self.k = k
        self.N = N

    def buildOneTree(self, df):
        data = selectData(df, self.classAttributeName, self.m, self.k)
        attributes = set(data.columns)
        return self.c45._buildTree(data, attributes.copy())

    def buildForest(self, df):
        self.trees = [self.buildOneTree(df) for _ in range(self.N)]

    def classify(self, row):
        classifiers = [Classifier(t) for t in self.trees]
        predictions = [
            classifier.classifySingle(classifier.T, row) for classifier in classifiers
        ]

        counts = {}
        mostCommon = predictions[0]
        for prediction in predictions:
            if counts.get(prediction) is None:
                counts[prediction] = 0
            counts[prediction] += 1
            if counts[prediction] > counts[mostCommon]:
                mostCommon = prediction
        return mostCommon

    def classifyMany(self, df):
        classifications = df.apply(self.classify, axis=1)
        return classifications
