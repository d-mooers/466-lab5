#!/usr/bin/env python3
# CSC 466
# Fall 2021
# Lab 3
# Dylan Mooers - dmooers@calpoly.edu
# Justin Huynh - jhuynh42@calpoly.edu
# Purpose - Runs C45

import pandas as pd
import numpy as np
import functools
import math


def getHomogenous(df, attributeName):
    values = df[attributeName].unique()
    if len(values) != 1:
        return None
    return values[0]


def mostCommonValue(df, attributeName):
    values = df[attributeName]
    occurences = {}
    maxOccurences = 0
    maxOccurenceName = ""
    for val in values:
        if occurences.get(val) is None:
            occurences[val] = 0
        occurences[val] += 1
        if occurences[val] > maxOccurences:
            maxOccurences = occurences[val]
            maxOccurenceName = val
    return (maxOccurences / len(df), maxOccurenceName)


def leaf(decision, p):
    return {"decision": decision, "p": p}


class Classifier:
    def __init__(self, T):
        self.T = T

    def classifySingle(self, node, dataRow):
        if node.get("decision") is not None:
            return node.get("decision")
        attribute = node.get("var")
        toTraverse = dataRow.loc[attribute]
        next = {}
        edges = node.get("edges")
        for edge in edges:
            if edge["edge"].get("direction") == "le" and float(toTraverse) <= float(
                edge["edge"]["value"]
            ):
                next = edge["edge"]
            elif edge["edge"].get("direction") == "gt" and float(toTraverse) > float(
                edge["edge"]["value"]
            ):
                next = edge["edge"]
            elif edge["edge"]["value"] == toTraverse:
                next = edge["edge"]
        if next.get("leaf"):
            return self.classifySingle(next["leaf"], dataRow)
        return self.classifySingle(next["node"], dataRow)

    def classify(self, df):
        def classifyWrapper(row):
            return self.classifySingle(self.T, row)

        classifications = df.apply(classifyWrapper, axis=1)
        return classifications


class C45:
    def __init__(
        self,
        data,
        classAttributeName,
        threshold,
        attributes,
        numericAttrs,
        useRatio=False,
    ):
        self.classAttributeValues = data[classAttributeName].unique()
        attributes.remove(classAttributeName)
        self.df = data
        self.root = None
        self.classAttributeName = classAttributeName
        self.threshold = threshold
        self.attributes = attributes
        self.numericAttrs = numericAttrs
        self.bestSplitNumber = -1
        self.useRatio = useRatio

    def calcPurity(self, df, attribute):
        return len(df[df[self.classAttributeName] == attribute]) / len(df)

    def calcEntropyInner(self, p):
        if p == 0:
            return 0
        return p * math.log2(p)

    def calcEntropy(self, df):
        return -1 * functools.reduce(
            lambda accum, x: accum + self.calcEntropyInner(self.calcPurity(df, x)),
            self.classAttributeValues,
            0,
        )

    def calcEntropyBinarySplit(self, splits, total):
        return -1 * functools.reduce(
            lambda accum, x: accum + self.calcEntropyInner(x / total), splits, 0
        )

    def calcInfoGainRatio(self, baseEntropy, df, splittingAttribute):
        values = df[splittingAttribute].unique()

        entropy = (
            functools.reduce(
                lambda accum, x: accum
                + self.calcEntropy(df[df[splittingAttribute] == x]),
                values,
                0,
            )
            / len(values)
        )
        # print(entropy)
        gain = baseEntropy - entropy
        if self.useRatio and entropy != 0:
            return gain / entropy
        return gain

    def infoGain(self, currentEntropy, df, a):
        if a in self.numericAttrs:
            return self.findBestSplit(a, df)
        return (self.calcInfoGainRatio(currentEntropy, df, a), -1)

    def selectSplittingAttribute(self, df, attributeSet):
        currentEntropy = self.calcEntropy(df)
        attributes = list(attributeSet)
        gainRatios = list(
            map(lambda a: self.infoGain(currentEntropy, df, a), attributes)
        )

        maxGainRatio = -1
        maxGainAttribute = ""
        maxSplittingNumber = -1
        for (gain, attribute) in list(zip(gainRatios, attributes)):
            # print(gain, attribute)
            gainRatio = gain[0]
            if gainRatio > maxGainRatio:
                maxGainRatio = gainRatio
                maxGainAttribute = attribute
                maxSplittingNumber = gain[1]
        if maxGainRatio < self.threshold:
            return None
        return (maxGainAttribute, maxSplittingNumber)

    def findBestSplit(self, Ai, df):
        possibleNumbers = sorted(df[Ai].unique())
        gain = 0
        maxGainSoFar = [-1, -1]
        counts = {
            num: {attribute: 0 for attribute in self.classAttributeValues}
            for num in possibleNumbers
        }
        totals = {num: 0 for num in possibleNumbers}
        antiCounts = {
            num: {attribute: 0 for attribute in self.classAttributeValues}
            for num in possibleNumbers
        }
        dfLen = len(df)

        p0 = self.calcEntropy(df)

        def vect(row):
            for num in possibleNumbers[: len(possibleNumbers) - 1]:
                if row[Ai] <= num:
                    counts[num][row[self.classAttributeName]] += 1
                    totals[num] += 1
                else:
                    antiCounts[num][row[self.classAttributeName]] += 1
            return row

        df.apply(vect, axis=1)

        for num in possibleNumbers[: len(possibleNumbers) - 1]:
            entropyLeft = (
                self.calcEntropyBinarySplit(counts[num].values(), totals[num])
                * totals[num]
                / dfLen
            )
            entropyRight = (
                self.calcEntropyBinarySplit(
                    antiCounts[num].values(), dfLen - totals[num]
                )
                * (dfLen - totals[num])
                / dfLen
            )
            entropy = entropyLeft + entropyRight
            # print(entropy, list(counts[num].values()), totals[num], Ai, num)
            gain = p0 - entropy
            if self.useRatio and entropy > 0:
                gain = gain / entropy
            if gain > maxGainSoFar[1]:
                maxGainSoFar[1] = gain
                maxGainSoFar[0] = num
        self.bestSplitNumber = maxGainSoFar[0]

        return (maxGainSoFar[1], maxGainSoFar[0])

    # TO-DO: Handle edge cases
    def _buildTree(self, df, attributes):
        homogenousAttribute = getHomogenous(df, self.classAttributeName)
        if not homogenousAttribute is None:
            return leaf(homogenousAttribute, 1)
        elif len(attributes) == 0:
            (maxOccurences, maxOccurenceName) = mostCommonValue(
                df, self.classAttributeName
            )
            return leaf(maxOccurenceName, maxOccurences)

        splittingAttr = self.selectSplittingAttribute(df, attributes)
        if splittingAttr is None:
            (maxOccurences, maxOccurenceName) = mostCommonValue(
                df, self.classAttributeName
            )
            return leaf(maxOccurenceName, maxOccurences)
        (splittingAttr, bestSplitNumber) = splittingAttr
        attributes.remove(splittingAttr)
        edges = []
        if splittingAttr in self.numericAttrs:
            leftNode = self._buildTree(
                df[df[splittingAttr] <= bestSplitNumber], attributes.copy()
            )
            rightNode = self._buildTree(
                df[df[splittingAttr] > bestSplitNumber], attributes.copy()
            )
            left = {"edge": {"value": bestSplitNumber, "direction": "le"}}
            right = {"edge": {"value": bestSplitNumber, "direction": "gt"}}

            if leftNode.get("decision"):
                left["edge"]["leaf"] = leftNode
            else:
                left["edge"]["node"] = leftNode

            if rightNode.get("decision"):
                right["edge"]["leaf"] = rightNode
            else:
                right["edge"]["node"] = rightNode
            edges.append(left)
            edges.append(right)
        else:
            edgeLabels = self.df[splittingAttr].unique()
            for edgeLabel in edgeLabels:
                split = df[df[splittingAttr] == edgeLabel]
                edge = {
                    "edge": {
                        "value": edgeLabel,
                    }
                }
                if len(split) > 0:
                    newNode = self._buildTree(split, attributes.copy())
                    if newNode.get("decision"):  # Must be leaf node
                        edge["edge"]["leaf"] = newNode
                    else:
                        edge["edge"]["node"] = newNode
                else:
                    (maxOccurences, maxOccurenceName) = mostCommonValue(
                        df, self.classAttributeName
                    )
                    edge["edge"]["leaf"] = leaf(maxOccurenceName, maxOccurences)

                edges.append(edge)
        return {"var": splittingAttr, "edges": edges}

    def buildTree(self, fileName):
        self.T = self._buildTree(self.df, self.attributes)
        return {"dataset": fileName, "node": self.T}
