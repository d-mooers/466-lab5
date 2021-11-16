import pandas as pd
import numpy as np
import argparse

def parse():
    parser = argparse.ArgumentParser(description="C45")
    parser.add_argument(
        "actual", type=str, help="Path to output of classifier"
    )
    parser.add_argument(
        "expected", type=str, help="Path to output of ground truth"
    )
    args = vars(parser.parse_args())
    return args

#!/usr/bin/env python3
# CSC 466
# Fall 2021
# Lab 3
# Dylan Mooers - dmooers@calpoly.edu
# Justin Huynh - jhuynh42@calpoly.edu
# Purpose - Calculate metrics and display results.

import argparse
import pandas as pd


def calcMetrics(groundTruth, predicitons, k=1):
    errors = 0
    correct = 0
    matrix = {}
    classNames = set()
    hits = {}
    misses = {}
    strikes = {}
    precision = {}
    recall = {}
    f1 = {}
    # isBinary = len(groundTruth.unique()) == 2
    # recall = None
    # precision = None
    # f1 = None

    for (actual, predicted) in zip(groundTruth, predicitons):
        if matrix.get(actual) is None:
            matrix[actual] = {}
        if matrix.get(actual).get(predicted) is None:
            matrix[actual][predicted] = 0
        matrix[actual][predicted] += 1
        classNames.add(actual)
        classNames.add(predicted)
        if actual == predicted:
            hits[actual] = hits.get(actual, 0) + 1
            correct += 1
        else:
            misses[predicted] = misses.get(predicted, 0) + 1
            strikes[actual] = strikes.get(actual, 0) + 1
            errors += 1
    metrics = {"author": [], "hits": [], "misses": [], "strikes": [], "recall": [], "precision": [], "f1": []}
    for name in classNames:
        if matrix.get(name) is None:
            matrix[name] = {}
        if matrix.get(name).get(name) is None:
            matrix[name][name] = 0
        recall[name] = hits.get(name, 0) / max((hits.get(name, 0) + misses.get(name, 0)), 1)
        precision[name] = hits.get(name, 0) / max((hits.get(name, 0) + strikes.get(name, 0)), 1)
        f1[name] = (2 * recall[name] * precision[name]) / max((recall[name] + precision[name]), 1)
        metrics['author'].append(name)
        metrics['recall'].append(recall[name])
        metrics['precision'].append(precision[name])
        metrics['f1'].append(f1[name])
        metrics['hits'].append(hits.get(name, 0))
        metrics['misses'].append(misses.get(name, 0))
        metrics['strikes'].append(strikes.get(name, 0))
        

    metrics = pd.DataFrame.from_dict(metrics)
    matrix = pd.DataFrame.from_dict(matrix)
    matrix.sort_index(axis=1, inplace=True)
    matrix.sort_index(axis=0, inplace=True)
    matrix.columns.name = "Actual"
    matrix.index.name = "Predicted"

    return (correct, errors, correct + errors, matrix.fillna(0), metrics.sort_values('f1', ascending=False))


def displayMetrics(
    data, results, classAttr, silent, k=1, displayTable=True, showResults=True
):
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.max_seq_items = None
    pd.options.display.width = 0

    (correct, errors, total, matrix, metrics) = calcMetrics(
        data[classAttr], results, k=k
    )
    if showResults:
        print("Confusion Matrix:\n")
        print(matrix)
        print("Total Correct: ", correct)
        print("Total Errors: ", errors)
        print("Accuracy: ", correct / total)
        print("Error Rate: ", errors / total)
        print("Metrics: ")
        print(metrics)

    return correct / total

def main():
    args = parse()
    actualPath = args['actual']
    expectedPath = args['expected']
    actual = pd.read_csv(actualPath)
    expected = pd.read_csv(expectedPath)
    displayMetrics(expected, actual['author'], 'author', False)

if __name__ == '__main__':
    main()